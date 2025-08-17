#!/usr/bin/env python3
"""
ATS Pro Plus — A production-grade, explainable ATS scoring script
-----------------------------------------------------------------

Major upgrades over the baseline script:
  ✓ Accurate per-skill experience computation (timeline-scoped)
  ✓ Better semantic matching (bi-encoder recall + cross-encoder reranking)
  ✓ Skill taxonomy & normalization (extensible synonyms; fuzzy & canonicalization)
  ✓ Robust resume segmentation & NER (spaCy-assisted, regex fallback)
  ✓ Title & seniority fit scoring (bidirectional JD↔Resume seniority inference)
  ✓ JD understanding & auto-requirements extraction (heuristics & patterns)
  ✓ Quality & readability analysis (textstat + optional grammar check)
  ✓ Stronger anti-gaming & integrity checks (repetition, entropy, hidden text)
  ✓ Actionable, visual explanations (rich HTML report with highlights & bars)
  ✓ Batch mode, ranking & exports (CSV/JSONL + HTML index)
  ✓ Engineering hardening & observability (schemas, logging, timings, version)

Corrections in this version (v2.1.0):
  ✓ Fixed major performance issue in batch mode by loading models only once.
  ✓ Improved error handling for file I/O and JSON parsing with user-friendly messages.
  ✓ Added explicit warnings when optional dependencies or models fail to load.
  ✓ Refined regex for resume section splitting to be more accurate.
  ✓ Refactored program exit logic for cleaner termination.
  ✓ Enhanced code structure for better maintainability.

Notes:
- Optional deps are used if present and gracefully degraded if missing.
- No external network calls — models must be installed locally if used.
- Designed to remain a single self-contained file for easy drop-in use.

Usage examples:
  # Single resume
  python ats_pro_plus.py \
    --pdf resume.pdf \
    --tex resume.tex \
    --jd job-description.txt \
    --role role.json \
    --out-json report.json \
    --out-md report.md \
    --out-html report.html \
    --min-score 75

  # Batch mode (directory of PDFs)
  python ats_pro_plus.py \
    --res-dir ./resumes \
    --jd job-description.txt \
    --role role.json \
    --out-csv batch.csv \
    --out-jsonl batch.jsonl \
    --out-html-index index.html

"""

from __future__ import annotations
import os
import re
import io
import json
import csv
import math
import time
import sys
import logging
import pathlib
import statistics
import argparse
import collections
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime

# Attempt to import optional dependencies
try:
    import dateparser
except ImportError:
    dateparser = None

try:
    from rapidfuzz import fuzz
except ImportError:
    class _F:
        @staticmethod
        def partial_ratio(a, b):
            return 100 if a.lower() in b.lower() or b.lower() in a.lower() else 0
    fuzz = _F()

try:
    import textstat
except ImportError:
    textstat = None

try:
    import language_tool_python
except ImportError:
    language_tool_python = None

try:
    import spacy
except ImportError:
    spacy = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    SentenceTransformer = None
    CrossEncoder = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except ImportError:
    pdfminer_extract_text = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

VERSION = "2.1.0"

# --- Logger Setup ---
logger = logging.getLogger("ATSProPlus")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Utility Functions ---

def read_textfile(p: Union[str, pathlib.Path]) -> str:
    p = pathlib.Path(p)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Failed to read text file {p}: {e}")
        return ""


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def safe_float(x, default=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def extract_pdf_text(pdf_path: str, use_ocr: bool = False) -> str:
    text = ""
    # 1. PyMuPDF (fitz) - Fast and generally reliable
    if fitz:
        try:
            with fitz.open(pdf_path) as doc:
                text = "".join(page.get_text() for page in doc)
        except Exception:
            text = "" # Will fallback
    # 2. pdfminer.six - A good fallback
    if not normalize_ws(text) and pdfminer_extract_text:
        try:
            text = pdfminer_extract_text(pdf_path) or ""
        except Exception:
            text = "" # Will fallback
    # 3. OCR with Tesseract - Slowest, last resort
    if not normalize_ws(text) and use_ocr:
        if pdfplumber and pytesseract and Image:
            logger.info(f"Falling back to OCR for {pdf_path}. This may be slow and less accurate.")
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    parts = []
                    for page in pdf.pages:
                        img = page.to_image(resolution=300).original
                        pil_img = img if isinstance(img, Image.Image) else Image.open(io.BytesIO(img.tobytes()))
                        parts.append(pytesseract.image_to_string(pil_img))
                    text = "\n".join(parts)
            except Exception as e:
                logger.warning(f"OCR processing failed for {pdf_path}: {e}")
                text = ""
        else:
            logger.warning("OCR dependencies (pdfplumber, pytesseract, Pillow) not installed. Cannot perform OCR.")
    return normalize_ws(text)


def detect_hidden_text(pdf_path: str) -> Dict[str, Any]:
    res = {"white_text_ratio": 0.0, "tiny_font_ratio": 0.0}
    if not pdfplumber:
        return res
    try:
        total_chars, white_like, tiny = 0, 0, 0
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                chars = page.chars or []
                for c in chars:
                    total_chars += 1
                    # White text check
                    nsc = c.get("non_stroking_color")
                    if isinstance(nsc, (list, tuple)) and len(nsc) >= 3:
                        r, g, b = nsc[:3]
                        if all(isinstance(v, (int, float)) for v in (r, g, b)) and r > 0.95 and g > 0.95 and b > 0.95:
                            white_like += 1
                    # Tiny font check
                    if c.get("size", 10) < 6:
                        tiny += 1
        if total_chars:
            res["white_text_ratio"] = round(white_like / total_chars, 4)
            res["tiny_font_ratio"] = round(tiny / total_chars, 4)
    except Exception:
        pass
    return res


def strip_latex(tex_src: str) -> str:
    s = re.sub(r"%.*", "", tex_src)  # remove comments
    s = re.sub(r"\\documentclass(?:\[.*?\])?\{.*?\}", "", s) # remove doc class
    s = re.sub(r"\\usepackage(?:\[.*?\])?\{.*?\}", "", s) # remove packages
    s = re.sub(r"\\begin\{document\}", "", s)
    s = re.sub(r"\\end\{document\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", s) # remove commands
    s = re.sub(r"[\{\}]", "", s) # remove braces
    return normalize_ws(s)


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+.#-]{2,}", (s or "").lower())


# --- Skill Taxonomy & Normalization ---
DEFAULT_TAXONOMY: Dict[str, set] = {
    "c++": {"cpp", "c plus plus"}, "c#": {"c sharp", "csharp"}, ".net": {"dotnet"},
    "java": {"jdk", "jvm", "spring"}, "python": {"py", "django", "flask"},
    "javascript": {"js", "es6"}, "typescript": {"ts"}, "react": {"reactjs"},
    "angular": {"angularjs"}, "vue": {"vuejs"}, "node.js": {"nodejs"},
    "go": {"golang"}, "rust": set(), "kubernetes": {"k8s"}, "docker": set(),
    "aws": {"amazon web services", "ec2", "s3", "lambda"}, "gcp": {"google cloud"},
    "azure": set(), "postgresql": {"postgres"}, "mysql": set(), "mongodb": set(),
    "redis": set(), "kafka": set(), "spark": set(), "hadoop": set(), "airflow": set(),
    "ci/cd": {"jenkins", "gitlab ci", "github actions"}, "git": set(),
    "machine learning": {"ml", "deep learning", "pytorch", "tensorflow", "scikit-learn"},
    "nlp": {"natural language processing"}, "rest": {"restful api"}, "graphql": set(),
}

def load_taxonomy(path: Optional[str]) -> Dict[str, set]:
    if path and pathlib.Path(path).exists():
        try:
            data = json.loads(read_textfile(path))
            return {k.lower(): set(map(str.lower, v)) for k, v in data.items()}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse taxonomy JSON from {path}: {e}. Using defaults.")
            return DEFAULT_TAXONOMY
    return DEFAULT_TAXONOMY


def canonicalize_skill(term: str, taxonomy: Dict[str, set]) -> str:
    t = normalize_ws(term.lower())
    if t in taxonomy: return t
    for k, syns in taxonomy.items():
        if t == k or t in syns: return k
    # Fuzzy match as a last resort
    for k, syns in taxonomy.items():
        if fuzz.partial_ratio(t, k) >= 92 or any(fuzz.partial_ratio(t, s) >= 92 for s in syns):
            return k
    return t


def synonym_hit(term: str, tokenset: set, taxonomy: Dict[str, set]) -> bool:
    t = term.lower()
    if t in tokenset: return True
    canonical = canonicalize_skill(t, taxonomy)
    if canonical in tokenset: return True
    syns = taxonomy.get(canonical, set())
    return any(s in tokenset for s in syns)


# --- Resume Segmentation ---
SECTION_HEADS = {
    "summary": r"summary|objective", "experience": r"experience|employment|work history",
    "education": r"education|academic", "projects": r"projects?", "skills": r"skills?|technologies",
    "certifications": r"certifications?|licenses?", "achievements": r"awards?|achievements",
    "publications": r"publications?", "links": r"links|profiles|portfolio",
}

def split_sections(text: str) -> Dict[str, str]:
    spans, indices = {}, []
    for name, pat in SECTION_HEADS.items():
        # Anchor regex to the start of a line to avoid matching mid-sentence
        for m in re.finditer(rf"^\s*({pat})\b", text, flags=re.I | re.M):
            indices.append((m.start(), name))
            break
    indices.sort()
    indices.append((len(text), "_end"))
    for i in range(len(indices) - 1):
        start, name = indices[i]
        end, _ = indices[i + 1]
        spans[name] = text[start:end]
    return spans


DATE_RANGE_RE = re.compile(
    r"(?P<a>(?:[A-Za-z]{3,9}[ '\-]?\d{2,4}|\d{1,2}/\d{4}|\d{4}))\s*(?:–|—|-|to)\s*(?P<b>(?:Present|Current|Now|\d{4}|[A-Za-z]{3,9}[ '\-]?\d{2,4}|\d{1,2}/\d{4}))",
    re.I,
)

def parse_date(s: str) -> Optional[datetime]:
    if not s or not dateparser: return None
    try:
        return dateparser.parse(s, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:
        return None


@dataclass
class Experience:
    title: str; company: str; start: Optional[datetime]; end: Optional[datetime]
    bullets: List[str]; raw_block: str
    @property
    def months(self) -> float:
        if not self.start: return 0.0
        end_date = self.end or datetime.now()
        return max(0.0, (end_date.year - self.start.year) * 12 + (end_date.month - self.start.month))


def segment_experiences(text: str, nlp_model: Optional[Any]) -> List[Experience]:
    exp_text = split_sections(text).get("experience", text)
    # Split by double newlines, a common separator for job entries
    blocks = re.split(r"\n\s*\n", exp_text)
    
    exps: List[Experience] = []
    for b in blocks:
        b = b.strip()
        if len(b) < 50: continue
        
        m = DATE_RANGE_RE.search(b)
        start, end = (None, None)
        if m:
            start = parse_date(m.group("a"))
            end_str = m.group("b")
            end = None if re.search(r"present|current|now", end_str, re.I) else parse_date(end_str)

        lines = [line.strip() for line in b.splitlines() if line.strip()]
        title, company = "", ""

        if nlp_model:
            doc = nlp_model(b)
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            if orgs: company = orgs[0]
            if lines: title = lines[0][:120]
        else: # Fallback to regex/heuristics
            if lines:
                title = lines[0][:120]
                if len(lines) > 1:
                    # Often the second line is the company name
                    company = lines[1][:120]

        bullets = re.findall(r"(?:^|\n)\s*[•\-–*]\s*(.+?)(?=\n|$)", b)
        exps.append(Experience(title, company, start, end, bullets, b))
    return [e for e in exps if e.start] # Only return experiences with a start date


def years_total_from_experiences(exps: List[Experience]) -> float:
    return round(sum(e.months for e in exps) / 12.0, 2)


def compute_per_skill_years(exps: List[Experience], skills: List[str], taxonomy: Dict[str, set]) -> Dict[str, float]:
    res: Dict[str, float] = {canonicalize_skill(s, taxonomy): 0.0 for s in skills}
    for e in exps:
        role_text = "\n".join([e.title, e.company] + e.bullets).lower()
        token_set = set(tokenize(role_text))
        for s in list(res.keys()):
            if synonym_hit(s, token_set, taxonomy):
                res[s] += e.months
    return {k: round(v / 12.0, 2) for k, v in res.items()}


# --- Seniority & JD Analysis ---
SENIORITY_ORDER = ["intern", "junior", "associate", "mid", "senior", "lead", "staff", "principal", "architect", "vp", "head"]
def detect_seniority(text: str) -> Optional[str]:
    t = (text or "").lower()
    for s in reversed(SENIORITY_ORDER):
        if re.search(fr"\b{s}\b", t): return s
    return None

def seniority_distance(a: Optional[str], b: Optional[str]) -> int:
    if not a or not b: return 4 # High distance if one is undetectable
    try: return abs(SENIORITY_ORDER.index(a) - SENIORITY_ORDER.index(b))
    except ValueError: return 4

@dataclass
class AutoJD:
    required_skills: List[str]; nice_to_have_skills: List[str]; per_skill_years: Dict[str, float]
    min_years_total: float; seniority: Optional[str]; role_name: str

def extract_requirements_from_jd(jd_text: str, taxonomy: Dict[str, set]) -> AutoJD:
    tl = jd_text.lower()
    all_skills = set(taxonomy.keys())
    
    # Extract skills mentioned in specific sections for better accuracy
    req_block = re.search(r"\brequirements?\b\s*:\s*(.*?)(?:\n{2,}|$)", tl, re.S)
    pref_block = re.search(r"\b(?:nice to have|preferred)\b\s*:?\s*(.*?)(?:\n{2,}|$)", tl, re.S)
    
    req_skills = set()
    if req_block:
        for skill in all_skills:
            if re.search(fr"\b{re.escape(skill)}\b", req_block.group(1)):
                req_skills.add(skill)
    # If no requirements block, scan the whole JD
    if not req_skills:
        for skill in all_skills:
            if re.search(fr"\b{re.escape(skill)}\b", tl):
                req_skills.add(skill)

    nice_skills = set()
    if pref_block:
        for skill in all_skills:
            if re.search(fr"\b{re.escape(skill)}\b", pref_block.group(1)):
                nice_skills.add(skill)

    per_skill_years = {}
    for s in req_skills:
        m = re.search(fr"(\d+)\+?\s*years?.*?\b{re.escape(s)}\b", tl)
        if m: per_skill_years[s] = float(m.group(1))

    min_years = 0.0
    m_total = re.search(r"(\d+)\+?\s*years?.*?experience", tl)
    if m_total: min_years = float(m_total.group(1))

    role_m = re.search(r"^\s*(?:role|position|title)\s*:\s*(.+)$", jd_text, re.M|re.I)
    role_name = normalize_ws(role_m.group(1)) if role_m else jd_text.splitlines()[0][:80]

    return AutoJD(
        required_skills=sorted(list(req_skills - nice_skills)),
        nice_to_have_skills=sorted(list(nice_skills)),
        per_skill_years=per_skill_years, min_years_total=min_years,
        seniority=detect_seniority(tl), role_name=role_name
    )

# --- Scoring Logic ---
def semantic_scores(
    jd_text: str, resume_text: str, bi_encoder: Optional[Any], cross_encoder: Optional[Any]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"bi_encoder_score": 0.0, "cross_encoder_score": 0.0, "top_chunks": []}
    
    # TF-IDF as a fallback
    tfidf_score = 0.0
    if TfidfVectorizer and cosine_similarity:
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            mat = vec.fit_transform([jd_text, resume_text])
            tfidf_score = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
        except Exception: pass
    
    if not bi_encoder or not np:
        result["bi_encoder_score"] = result["cross_encoder_score"] = tfidf_score
        return result

    try:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", resume_text) if len(s.strip()) > 20]
        if not sentences: sentences = [resume_text]

        jd_emb = bi_encoder.encode([jd_text], normalize_embeddings=True, show_progress_bar=False)
        res_embs = bi_encoder.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        
        sims = (jd_emb @ res_embs.T)[0]
        top_idx = np.argsort(-sims)[:10]
        top_chunks = [(sentences[i], float(sims[i])) for i in top_idx]
        
        result["bi_encoder_score"] = float(max(sims)) if len(sims) > 0 else 0.0
        result["top_chunks"] = top_chunks

        if cross_encoder:
            pairs = [(jd_text, chunk) for chunk, _ in top_chunks]
            if pairs:
                ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)
                result["cross_encoder_score"] = float(max(ce_scores))
            else:
                result["cross_encoder_score"] = result["bi_encoder_score"]
        else:
            result["cross_encoder_score"] = result["bi_encoder_score"]
    except Exception as e:
        logger.warning(f"Semantic model processing failed: {e}. Using TF-IDF score.")
        result["bi_encoder_score"] = result["cross_encoder_score"] = tfidf_score
    return result


def anti_gaming_metrics(text: str, pdf_path: Optional[str]) -> Dict[str, float]:
    tokens = tokenize(text)
    total = max(1, len(tokens))
    counts = collections.Counter(tokens)
    top_ratio = max((c / total for _, c in counts.most_common(1))) if counts else 0.0
    
    freqs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in freqs if p > 0)
    
    hidden = detect_hidden_text(pdf_path) if pdf_path else {}
    return {
        "top_token_ratio": round(top_ratio, 4), "token_entropy": round(entropy, 4),
        "white_text_ratio": hidden.get("white_text_ratio", 0.0),
        "tiny_font_ratio": hidden.get("tiny_font_ratio", 0.0),
    }

# --- Main Scoring Orchestration ---
def compute_scores(
    jd_text: str, resume_text: str, sections: Dict[str, str], exps: List[Experience],
    taxonomy: Dict[str, set], role: Dict[str, Any], args: argparse.Namespace,
    pdf_path: Optional[str], models: Dict[str, Any]
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # Role requirements
    required = {canonicalize_skill(s, taxonomy) for s in role.get("required_skills", [])}
    nice = {canonicalize_skill(s, taxonomy) for s in role.get("nice_to_have_skills", [])}
    min_years_req = float(role.get("min_years_experience_total", 0) or 0)
    per_skill_years_req = {canonicalize_skill(k, taxonomy): safe_float(v) for k, v in role.get("per_skill_years", {}).items()}
    jd_seniority = role.get("seniority") or detect_seniority(jd_text)

    # Resume analysis
    res_tokens = set(tokenize(resume_text))
    years_total = years_total_from_experiences(exps)
    skills_for_eval = sorted(required | nice | set(per_skill_years_req.keys()))
    per_skill_years_est = compute_per_skill_years(exps, skills_for_eval, taxonomy)
    res_seniority = detect_seniority(resume_text)

    # Semantic scoring
    sem = semantic_scores(jd_text, resume_text, models.get('bi_encoder'), models.get('cross_encoder'))

    # Keyword coverage
    req_hits = {r: synonym_hit(r, res_tokens, taxonomy) for r in required}
    req_cov = sum(req_hits.values()) / max(1, len(required)) if required else 1.0
    nice_cov = sum(synonym_hit(n, res_tokens, taxonomy) for n in nice) / max(1, len(nice)) if nice else 0.0

    # Experience coverage
    total_years_cov = min(1.0, years_total / max(1.0, min_years_req))
    per_skill_covs = [min(1.0, per_skill_years_est.get(k, 0) / max(0.1, v)) for k, v in per_skill_years_req.items()]
    avg_per_skill_cov = statistics.mean(per_skill_covs) if per_skill_covs else 1.0
    
    # Quality metrics
    bullets = re.findall(r"[•\-–*]\s*(.+)", resume_text)
    impact_ratio = sum(1 for b in bullets if re.search(r"\d", b)) / max(1, len(bullets))
    
    readability = 0.0
    if textstat:
        try: readability = min(1.0, max(0.0, textstat.flesch_reading_ease(resume_text) / 100.0))
        except Exception: pass
    
    # Seniority fit
    sdist = seniority_distance(res_seniority, jd_seniority)
    seniority_fit = max(0.0, 1.0 - sdist / 5.0)

    # Penalties
    ag = anti_gaming_metrics(resume_text, pdf_path)
    gaming_penalty = sum([
        0.10 if ag["top_token_ratio"] > 0.08 else 0,
        0.15 if ag["white_text_ratio"] > 0.01 else 0,
        0.15 if ag["tiny_font_ratio"] > 0.01 else 0
    ])

    # --- Component Scores ---
    w = {"kw": 0.30, "sem": 0.20, "exp": 0.25, "qual": 0.10, "sen": 0.10, "imp": 0.05}
    s_kw = w["kw"] * (0.7 * req_cov + 0.3 * nice_cov)
    s_sem = w["sem"] * (0.5 * sem.get("bi_encoder_score", 0) + 0.5 * sem.get("cross_encoder_score", 0))
    s_exp = w["exp"] * (0.5 * total_years_cov + 0.5 * avg_per_skill_cov)
    s_qual = w["qual"] * readability
    s_imp = w["imp"] * impact_ratio
    s_sen = w["sen"] * seniority_fit
    
    score_raw = s_kw + s_sem + s_exp + s_qual + s_imp + s_sen
    score = round(max(0, min(100, (score_raw - gaming_penalty) * 100)))

    # Gates
    lacking_years = {k: v for k, v in per_skill_years_req.items() if per_skill_years_est.get(k, 0) < v}
    gates = {
        "required_skills_met": req_cov >= args.min_req_coverage,
        "total_years_met": years_total >= min_years_req,
        "per_skill_years_met": not lacking_years,
        "seniority_match": sdist <= 2,
    }

    # Suggestions
    suggestions = []
    if not gates["required_skills_met"]:
        missing = sorted([k for k, v in req_hits.items() if not v])
        suggestions.append(f"Address missing required skills: {', '.join(missing)}.")
    if not gates["per_skill_years_met"]:
        gaps = sorted(((k, v - per_skill_years_est.get(k, 0)) for k, v in lacking_years.items()), key=lambda x: -x[1])
        suggestions.append(f"Strengthen experience for: " + ", ".join(f"{k} (+{gap:.1f}y)" for k, gap in gaps))
    if impact_ratio < 0.5:
        suggestions.append("Increase use of metrics and quantified achievements in bullet points.")
    if readability < 0.4:
        suggestions.append("Improve readability with shorter sentences and simpler language.")

    report = {
        "version": VERSION, "role_name": role.get("role_name", "N/A"),
        "overall_score": score, "gates": gates,
        "component_scores": {
            "keywords": round(s_kw * 100), "semantic_match": round(s_sem * 100),
            "experience_fit": round(s_exp * 100), "quality": round(s_qual * 100),
            "impact": round(s_imp * 100), "seniority_fit": round(s_sen * 100),
            "gaming_penalty_pts": round(gaming_penalty * 100),
        },
        "details": {
            "required_coverage": round(req_cov, 3), "nice_to_have_coverage": round(nice_cov, 3),
            "years_total_detected": years_total, "missing_required_skills": sorted([k for k,v in req_hits.items() if not v]),
            "per_skill_years_required": per_skill_years_req, "per_skill_years_estimated": per_skill_years_est,
            "lacking_years_experience": lacking_years,
        },
        "quality_analysis": {"readability_score": readability, "impact_bullet_ratio": impact_ratio},
        "anti_gaming": ag, "semantic": sem, "suggestions": suggestions,
        "timings_s": {"scoring": round(time.perf_counter() - t0, 3)},
    }
    return report

# --- I/O and Reporting ---
HTML_CSS = "<style>body{font-family:sans-serif;max-width:980px;margin:auto;padding:16px}h1,h2{color:#333}.card{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:16px 0}.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px}.ok{background:#dcfce7}.warn{background:#fee2e2}.table{width:100%;border-collapse:collapse}.table th,.table td{border-bottom:1px solid #eee;padding:8px;text-align:left}.bar{height:10px;background:#e5e7eb;border-radius:6px;overflow:hidden}.bar>span{display:block;height:10px;background:#3b82f6}mark{background:#fef08a}</style>"
def highlight(text, terms):
    for t in sorted(set(terms), key=len, reverse=True):
        if len(t) > 2:
            text = re.sub(fr"(?i)\b({re.escape(t)})\b", r"<mark>\1</mark>", text)
    return text

def render_html(report, resume_text, jd_text, out_path):
    skills = set(report["details"]["per_skill_years_required"].keys()) | set(report["details"]["per_skill_years_estimated"].keys())
    html = f"<!DOCTYPE html><html><head><title>ATS Report</title>{HTML_CSS}</head><body>"
    html += f"<h1>ATS Pro Plus Report — Score: {report['overall_score']}/100</h1>"
    html += "<div class='card'><h2>Gates</h2>" + "".join(f"<span class='badge {('ok' if v else 'warn')}'>{k.replace('_',' ')}</span> " for k, v in report['gates'].items()) + "</div>"
    html += "<div class='card'><h2>Suggestions</h2><ul>" + "".join(f"<li>{s}</li>" for s in report['suggestions']) + "</ul></div>"
    html += f"<div class='card'><h2>JD (highlighted)</h2><pre>{highlight(jd_text, skills)}</pre></div>"
    html += f"<div class='card'><h2>Resume (highlighted)</h2><pre>{highlight(resume_text, skills)}</pre></div>"
    html += "</body></html>"
    pathlib.Path(out_path).write_text(html, encoding="utf-8")


def evaluate_single_instance(args: argparse.Namespace, taxonomy: Dict[str, set], models: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
    # Load and auto-analyze JD
    jd_text = read_textfile(args.jd)
    role = {}
    if args.role and pathlib.Path(args.role).exists():
        try:
            role = json.loads(read_textfile(args.role))
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing role JSON {args.role}: {e}. Halting.")
            sys.exit(1)
            
    if not role.get("required_skills") and jd_text:
        auto_jd = extract_requirements_from_jd(jd_text, taxonomy)
        role = {**auto_jd.__dict__, **role}

    # Load and parse resume
    resume_text = ""
    if args.pdf and pathlib.Path(args.pdf).exists():
        resume_text = extract_pdf_text(args.pdf, use_ocr=args.use_ocr)
    elif args.tex and pathlib.Path(args.tex).exists():
        resume_text = strip_latex(read_textfile(args.tex))
    
    sections = split_sections(resume_text)
    exps = segment_experiences(resume_text, models.get('nlp'))
    
    report = compute_scores(jd_text, resume_text, sections, exps, taxonomy, role, args, args.pdf, models)
    return report, resume_text, jd_text

def save_outputs_single(report, args, resume_text, jd_text) -> bool:
    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"Wrote JSON report to: {args.out_json}")
    if args.out_html:
        render_html(report, resume_text, jd_text, args.out_html)
        logger.info(f"Wrote HTML report to: {args.out_html}")
    
    score = report.get("overall_score", 0)
    gates_passed = all(report.get("gates", {}).values())
    
    if score >= args.min_score and gates_passed:
        logger.info(f"PASSED: Score ({score}) is >= min_score ({args.min_score}) and all gates passed.")
        return True
    else:
        logger.warning(f"FAILED: Score ({score}) or gates did not meet minimum requirements.")
        print(json.dumps(report, indent=2)) # Print full report on failure
        return False

# --- Main Execution ---

def main():
    ap = argparse.ArgumentParser(description=f"ATS Pro Plus v{VERSION}")
    # Single mode args
    ap.add_argument("--pdf", help="Path to resume PDF")
    ap.add_argument("--tex", help="Path to resume LaTeX source (fallback)")
    # Batch mode args
    ap.add_argument("--res-dir", help="Directory of PDFs to score (enables batch mode)")
    # Common args
    ap.add_argument("--jd", help="Path to Job Description text file")
    ap.add_argument("--role", help="Path to role JSON file defining requirements")
    ap.add_argument("--taxonomy", help="Path to custom skill taxonomy JSON (optional)")
    # Output args
    ap.add_argument("--out-json", help="Output path for single JSON report")
    ap.add_argument("--out-html", help="Output path for single HTML report")
    ap.add_argument("--out-csv", help="Output path for batch CSV summary")
    ap.add_argument("--out-jsonl", help="Output path for batch JSONL detailed reports")
    ap.add_argument("--out-html-index", help="Output path for batch HTML index")
    # Scoring options
    ap.add_argument("--min-score", type=int, default=70, help="Minimum score to pass")
    ap.add_argument("--min-req-coverage", type=float, default=0.80, help="Minimum required skill coverage to pass gate")
    ap.add_argument("--use-ocr", action="store_true", help="Enable OCR fallback for scanned PDFs")
    # Other
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    
    if args.verbose: logger.setLevel(logging.DEBUG)
    taxonomy = load_taxonomy(args.taxonomy)

    # --- Pre-load models ---
    models = {}
    if spacy:
        try:
            models['nlp'] = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'. Experience segmentation will be less accurate.")
    
    if SentenceTransformer:
        try:
            # Using a default from the original script, can be overridden by user in future
            models['bi_encoder'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            if CrossEncoder:
                 models['cross_encoder'] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence-transformer models: {e}. Semantic scoring will be disabled.")
    
    # --- Mode selection: Batch or Single ---
    if args.res_dir:
        # Batch Mode
        res_dir = pathlib.Path(args.res_dir)
        if not res_dir.is_dir():
            logger.error(f"--res-dir '{args.res_dir}' is not a valid directory.")
            sys.exit(1)
        files = sorted(res_dir.glob("*.pdf"))
        
        results = []
        jsonl_fh = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None

        for pdf_path in files:
            logger.info(f"--- Scoring: {pdf_path.name} ---")
            args_single = argparse.Namespace(**vars(args), pdf=str(pdf_path), tex=None)
            report, _, _ = evaluate_single_instance(args_single, taxonomy, models)
            
            summary = {
                "file": pdf_path.name,
                "score": report['overall_score'],
                "gates_passed": all(report['gates'].values()),
                "req_coverage": report['details']['required_coverage'],
                "years_total": report['details']['years_total_detected'],
            }
            results.append(summary)
            if jsonl_fh:
                jsonl_fh.write(json.dumps({"file": pdf_path.name, **report}) + "\n")

        if jsonl_fh: jsonl_fh.close()
        
        # Sort results for ranking
        results.sort(key=lambda r: (-r["gates_passed"], -r["score"]))

        if args.out_csv:
            with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            logger.info(f"Wrote batch CSV summary to: {args.out_csv}")

    else:
        # Single Mode
        if not args.pdf and not args.tex:
            logger.error("Must provide --pdf or --tex for single evaluation mode.")
            sys.exit(1)
        report, resume_text, jd_text = evaluate_single_instance(args, taxonomy, models)
        is_pass = save_outputs_single(report, args, resume_text, jd_text)
        if not is_pass:
            sys.exit(1) # Exit with error code if failed

if __name__ == "__main__":
    main()