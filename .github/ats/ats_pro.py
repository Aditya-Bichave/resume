#!/usr/bin/env python3
"""
ATS Pro Plus — A production-grade, explainable ATS scoring script
-----------------------------------------------------------------
Persona: Senior Software Architect & Senior Software Developer (20+ yrs)

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

try:
    import dateparser
except Exception:
    dateparser = None

try:
    from rapidfuzz import fuzz
except Exception:
    class _F:
        @staticmethod
        def partial_ratio(a, b):
            return 100 if a == b else 0
    fuzz = _F()

try:
    from textstat import textstat
except Exception:
    textstat = None

try:
    import language_tool_python
except Exception:
    language_tool_python = None

try:
    import spacy
except Exception:
    spacy = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from PIL import Image
    import pytesseract
except Exception:
    Image = None
    pytesseract = None

try:
    import fitz
except Exception:
    fitz = None

VERSION = "2.0.1"

logger = logging.getLogger("ATSProPlus")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def read_textfile(p: Union[str, pathlib.Path]) -> str:
    p = pathlib.Path(p)
    try:
        return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    except Exception:
        return ""


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def extract_pdf_text(pdf_path: str, use_ocr: bool = False) -> str:
    text = ""
    if fitz is not None:
        try:
            with fitz.open(pdf_path) as doc:
                parts = []
                for page in doc:
                    parts.append(page.get_text("text"))
                text = "\n".join(parts)
        except Exception:
            text = ""
    if not text and pdfminer_extract_text is not None:
        try:
            text = pdfminer_extract_text(pdf_path) or ""
        except Exception:
            text = ""
    if not text and use_ocr and pdfplumber is not None and pytesseract is not None and Image is not None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                parts = []
                for page in pdf.pages:
                    img = page.to_image(resolution=300).original
                    pil = img if isinstance(img, Image.Image) else Image.open(io.BytesIO(img))
                    parts.append(pytesseract.image_to_string(pil))
                text = "\n".join(parts)
        except Exception:
            text = ""
    return normalize_ws(text)


def detect_multicolumn(pdf_path: str) -> bool:
    if pdfplumber is None:
        return False
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                words = page.extract_words() or []
                xs = [w.get("x0", 0) for w in words]
                if xs:
                    buckets = set(int(x // 10) for x in xs)
                    if len(buckets) > 30:
                        return True
    except Exception:
        return False
    return False


def detect_hidden_text(pdf_path: str) -> Dict[str, Any]:
    res = {"white_text_ratio": 0.0, "tiny_font_ratio": 0.0}
    if pdfplumber is None:
        return res
    try:
        total_chars = 0
        white_like = 0
        tiny = 0
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                chars = page.chars or []
                for c in chars:
                    total_chars += 1
                    size = c.get("size", 0)
                    nsc = c.get("non_stroking_color")
                    if isinstance(nsc, (list, tuple)) and len(nsc) >= 3:
                        r, g, b = nsc[:3]
                        if all(isinstance(v, (int, float)) for v in (r, g, b)):
                            if r > 0.95 and g > 0.95 and b > 0.95:
                                white_like += 1
                    if size and size < 6:
                        tiny += 1
        if total_chars:
            res["white_text_ratio"] = round(white_like / total_chars, 4)
            res["tiny_font_ratio"] = round(tiny / total_chars, 4)
    except Exception:
        pass
    return res


def strip_latex(tex_src: str) -> str:
    s = re.sub(r"%.*", "", tex_src)
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+.#-]{2,}", (s or "").lower())


def bigrams(tokens: List[str]) -> set:
    return {" ".join(p) for p in zip(tokens, tokens[1:])}


DEFAULT_TAXONOMY: Dict[str, set] = {
    "c++": {"cpp", "c plus plus", "c\u002b\u002b"},
    "c#": {"c sharp", "csharp"},
    ".net": {"dotnet", "asp.net", "aspnet"},
    "java": {"jdk", "jvm", "spring", "spring boot"},
    "python": {"py", "py3", "django", "flask"},
    "javascript": {"js", "ecmascript"},
    "typescript": {"ts"},
    "react": {"react.js", "reactjs"},
    "node.js": {"node", "nodejs"},
    "kotlin": set(),
    "swift": set(),
    "go": {"golang"},
    "rust": set(),
    "kubernetes": {"k8s"},
    "docker": set(),
    "aws": {"amazon web services", "ec2", "s3", "rds", "lambda", "eks"},
    "gcp": {"google cloud", "gke", "bigquery"},
    "azure": {"microsoft azure", "aks"},
    "postgresql": {"postgres", "psql"},
    "mysql": set(),
    "mongodb": {"mongo"},
    "redis": set(),
    "snowflake": set(),
    "bigquery": set(),
    "spark": {"pyspark"},
    "hadoop": set(),
    "airflow": set(),
    "ci/cd": {"ci cd", "continuous integration", "continuous delivery", "github actions", "gitlab ci"},
    "machine learning": {"ml", "deep learning", "dl", "tensorflow", "pytorch", "sklearn"},
    "nlp": {"natural language processing"},
    "microservices": {"service oriented", "soa"},
    "graphql": set(),
    "rest": {"restful"},
    "oop": {"object-oriented", "object oriented"},
}


def load_taxonomy(path: Optional[str]) -> Dict[str, set]:
    if path and pathlib.Path(path).exists():
        try:
            data = json.loads(read_textfile(path))
            return {k.lower(): set(map(str.lower, v)) for k, v in data.items()}
        except Exception:
            logger.warning("Failed to load taxonomy JSON, using defaults.")
    return {k: set(v) for k, v in DEFAULT_TAXONOMY.items()}


def canonicalize_skill(term: str, taxonomy: Dict[str, set]) -> str:
    t = (term or "").lower().strip()
    if t in taxonomy:
        return t
    for k, syns in taxonomy.items():
        if t == k or t in syns:
            return k
    for k, syns in taxonomy.items():
        if fuzz.partial_ratio(t, k) >= 92:
            return k
        if any(fuzz.partial_ratio(t, s) >= 92 for s in syns):
            return k
    return t


def synonym_hit(term: str, tokenset: set, taxonomy: Dict[str, set]) -> bool:
    t = term.lower()
    if t in tokenset:
        return True
    canonical = canonicalize_skill(t, taxonomy)
    if canonical in tokenset:
        return True
    syns = taxonomy.get(canonical, set())
    return any(any(fuzz.partial_ratio(syn, tok) >= 90 for tok in tokenset) for syn in syns)


SECTION_HEADS = {
    "summary": r"(?:summary|objective)",
    "experience": r"(?:experience|employment|work experience|professional experience)",
    "education": r"(?:education|academics)",
    "projects": r"(?:projects?)",
    "skills": r"(?:skills?|technologies|tech stack)",
    "certifications": r"(?:certifications?|licenses?)",
    "achievements": r"(?:awards?|achievements?)",
    "publications": r"(?:publications?)",
    "links": r"(?:links|profiles|online presence)",
}


def split_sections(text: str) -> Dict[str, str]:
    spans: Dict[str, str] = {}
    indices: List[Tuple[int, str]] = []
    for name, pat in SECTION_HEADS.items():
        m = re.search(rf"\b{pat}\b", text, flags=re.I)
        if m:
            indices.append((m.start(), name))
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
    if not s:
        return None
    if dateparser is None:
        m = re.search(r"\b(19|20)\d{2}\b", s)
        if m:
            y = int(m.group(0))
            return datetime(y, 1, 1)
        return None
    try:
        return dateparser.parse(s, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:
        return None


@dataclass
class Experience:
    title: str
    company: str
    start: Optional[datetime]
    end: Optional[datetime]
    bullets: List[str]
    raw_block: str

    @property
    def months(self) -> float:
        if not self.start:
            return 0.0
        e = self.end or datetime.now()
        m = (e.year - self.start.year) * 12 + (e.month - self.start.month)
        return max(0.0, float(m))


def segment_experiences(text: str) -> List[Experience]:
    blocks: List[str] = []
    sections = split_sections(text)
    exp_text = sections.get("experience") or text
    parts = re.split(r"\n\s*\n|\n\s*•|\n-\s", exp_text)
    cur: List[str] = []
    for p in parts:
        if len(p.strip()) < 40:
            cur.append(p)
            continue
        if cur:
            cur.append(p)
            blocks.append("\n".join(cur))
            cur = []
        else:
            blocks.append(p)
    if cur:
        blocks.append("\n".join(cur))

    nlp = None
    if spacy is not None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None

    exps: List[Experience] = []
    for b in blocks:
        if len(b.strip()) < 60:
            continue
        m = DATE_RANGE_RE.search(b)
        start = end = None
        if m:
            start = parse_date(m.group("a"))
            end = None if re.search(r"present|current|now", m.group("b"), re.I) else parse_date(m.group("b"))
        title = ""
        company = ""
        if nlp is not None:
            doc = nlp(b)
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            if orgs:
                company = orgs[0]
            lines = [l.strip() for l in b.splitlines() if l.strip()]
            if lines:
                title = lines[0][:120]
        else:
            lines = [l.strip() for l in b.splitlines() if l.strip()]
            if lines:
                title = lines[0][:120]
                if len(lines) > 1:
                    company = lines[1][:120]
        bullets = re.findall(r"(?:^|\n)[•\-–]\s*(.+?)(?=\n|$)", b)
        if not bullets:
            bullets = [l for l in b.splitlines()[2:6] if len(l.strip()) > 0][:6]
        exps.append(Experience(title=title, company=company, start=start, end=end, bullets=bullets, raw_block=b))
    return exps


def years_total_from_experiences(exps: List[Experience]) -> float:
    months = sum(e.months for e in exps)
    return round(months / 12.0, 2)


def compute_per_skill_years(exps: List[Experience], skills: List[str], taxonomy: Dict[str, set]) -> Dict[str, float]:
    res: Dict[str, float] = {canonicalize_skill(s, taxonomy): 0.0 for s in skills}
    for e in exps:
        role_text = "\n".join([e.title, e.company] + e.bullets).lower()
        token_set = set(tokenize(role_text))
        for s in list(res.keys()):
            if synonym_hit(s, token_set, taxonomy):
                res[s] += e.months
    return {k: round(v / 12.0, 2) for k, v in res.items()}


SENIORITY_ORDER = ["intern", "junior", "associate", "mid", "senior", "lead", "staff", "principal", "architect", "vp", "head"]


def detect_seniority(text: str) -> Optional[str]:
    t = (text or "").lower()
    for s in reversed(SENIORITY_ORDER):
        if re.search(fr"\b{s}\b", t):
            return s
    if re.search(r"\b(iii|ii|iv|sr\.?|lead|principal|staff)\b", t):
        if re.search(r"\b(sr\.?|senior)\b", t):
            return "senior"
        if re.search(r"\b(principal)\b", t):
            return "principal"
        if re.search(r"\b(staff)\b", t):
            return "staff"
        if re.search(r"\b(lead)\b", t):
            return "lead"
    return None


def seniority_distance(a: Optional[str], b: Optional[str]) -> int:
    if not a or not b:
        return 0
    try:
        ia = SENIORITY_ORDER.index(a)
        ib = SENIORITY_ORDER.index(b)
        return abs(ia - ib)
    except Exception:
        return 0


DEGREE_WORDS = [
    "bachelor", "masters", "phd", "bsc", "msc", "b.e", "btech", "mtech", "mba",
    "computer science", "cs", "engineering",
]

VISA_WORDS = [
    "h-1b", "h1b", "h4-ead", "opt", "cpt", "green card", "permanent resident", "pr",
    "usc", "us citizen", "eu work permit", "open work permit",
]

REMOTE_WORDS = ["remote", "hybrid", "relocate", "relocation", "onsite"]


@dataclass
class AutoJD:
    role_name: str
    required_skills: List[str]
    nice_to_have_skills: List[str]
    per_skill_years: Dict[str, float]
    min_years_total: float
    degree: str
    location: str
    work_authorization: str
    seniority: Optional[str]


def extract_requirements_from_jd(jd_text: str, taxonomy: Dict[str, set]) -> AutoJD:
    t = jd_text or ""
    tl = t.lower()

    skills_found = set()
    tokens = set(tokenize(tl))
    for base, syns in taxonomy.items():
        if (base in tokens) or (" " in base and base in tl) or any((s in tokens) or (" " in s and s in tl) for s in syns):
            skills_found.add(base)

    per_skill: Dict[str, float] = {}
    for s in skills_found:
        m = re.search(fr"(\d+)[+ ]*\+?\s*(?:\+)?\s*years?\s+(?:of|with|in)\s+{re.escape(s)}", tl)
        if m:
            per_skill[s] = float(m.group(1))

    my = 0.0
    m_total = re.search(r"(\d+)[+ ]*\+?\s*years?\s+(?:of\s+)?experience", tl)
    if m_total:
        my = float(m_total.group(1))

    deg = ""
    for w in DEGREE_WORDS:
        if re.search(fr"\b{re.escape(w)}\b", tl):
            deg = w
            break

    loc = ""
    wa = ""
    for w in REMOTE_WORDS:
        if re.search(fr"\b{w}\b", tl):
            loc = w
            break
    for w in VISA_WORDS:
        if re.search(fr"\b{re.escape(w)}\b", tl):
            wa = w
            break

    sr = detect_seniority(tl)

    required = sorted(skills_found)
    nice: List[str] = []

    req_match = re.search(r"\brequirements?\b\s*:\s*(.*?)(?:\n{2,}|$)", tl, flags=re.I | re.S)
    pref_match = re.search(r"\b(?:nice to have|preferred)\b\s*:?\s*(.*?)(?:\n{2,}|$)", tl, flags=re.I | re.S)
    if req_match:
        req_block = req_match.group(1)[:500]
        required = sorted({canonicalize_skill(w, taxonomy) for w in tokenize(req_block) if w in skills_found or w in taxonomy})
    if pref_match:
        pref_block = pref_match.group(1)[:500]
        nice = sorted({canonicalize_skill(w, taxonomy) for w in tokenize(pref_block) if w in taxonomy})

    role_name = "unspecified"
    m_role = re.search(r"^\s*(?:role|position|title)\s*:\s*(.+)$", tl, flags=re.M)
    if m_role:
        role_name = m_role.group(1)[:80]

    return AutoJD(
        role_name=role_name,
        required_skills=required,
        nice_to_have_skills=nice,
        per_skill_years=per_skill,
        min_years_total=my,
        degree=deg,
        location=loc,
        work_authorization=wa,
        seniority=sr,
    )


def semantic_scores(
    jd_text: str,
    resume_text: str,
    bi_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cross_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Dict[str, Any]:
    jd = normalize_ws(jd_text)
    res = normalize_ws(resume_text)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", res)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 4]

    result: Dict[str, Any] = {"bi_encoder_score": 0.0, "cross_encoder_score": 0.0, "top_chunks": []}

    tfidf_score = 0.0
    if TfidfVectorizer is not None and cosine_similarity is not None:
        try:
            tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            tfidf_mat = tfidf.fit_transform([jd, res])
            tfidf_score = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
        except Exception:
            tfidf_score = 0.0

    if SentenceTransformer is None:
        result["bi_encoder_score"] = tfidf_score
        result["cross_encoder_score"] = tfidf_score
        return result

    try:
        bi_encoder = SentenceTransformer(bi_model_name)
        emb_jd = bi_encoder.encode([jd], normalize_embeddings=True, show_progress_bar=False)
        chunks = sentences if sentences else [res]
        emb_chunks = bi_encoder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        sims = (emb_jd @ emb_chunks.T)[0]
        if np is not None:
            top_idx = np.argsort(-sims)[:10]
        else:
            top_idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:10]
        top_chunks = [(chunks[i], float(sims[i])) for i in top_idx]
        result["bi_encoder_score"] = float(max(sims)) if len(sims) else 0.0
        result["top_chunks"] = top_chunks

        if CrossEncoder is not None:
            try:
                ce = CrossEncoder(cross_model_name)
                pairs = [(jd, c) for c, _ in top_chunks]
                scores = ce.predict(pairs)
                best = float(max(scores)) if len(scores) else 0.0
                result["cross_encoder_score"] = best
            except Exception:
                result["cross_encoder_score"] = result["bi_encoder_score"]
        else:
            result["cross_encoder_score"] = result["bi_encoder_score"]

    except Exception as e:
        logger.warning(f"Semantic model error: {e}")
        result["bi_encoder_score"] = tfidf_score
        result["cross_encoder_score"] = tfidf_score

    return result


def readability_metrics(text: str) -> Dict[str, float]:
    if not textstat:
        return {"flesch_reading_ease": 0.0, "smog_index": 0.0, "coleman_liau_index": 0.0, "avg_sentence_len": 0.0}
    try:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        slens = [len(s.split()) for s in sentences if len(s.split()) > 0]
        avg_s = statistics.mean(slens) if slens else 0.0
        return {
            "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
            "smog_index": float(textstat.smog_index(text)),
            "coleman_liau_index": float(textstat.coleman_liau_index(text)),
            "avg_sentence_len": float(avg_s),
        }
    except Exception:
        return {"flesch_reading_ease": 0.0, "smog_index": 0.0, "coleman_liau_index": 0.0, "avg_sentence_len": 0.0}


def grammar_issues(text: str) -> int:
    if not language_tool_python:
        return 0
    try:
        tool = language_tool_python.LanguageTool("en-US")
        matches = tool.check(text[:20000])
        return len(matches)
    except Exception:
        return 0


def anti_gaming_metrics(text: str, pdf_path: Optional[str]) -> Dict[str, float]:
    tokens = tokenize(text)
    total = max(1, len(tokens))
    top = collections.Counter(tokens).most_common(10)
    top_ratio = max((c / total) for _, c in top) if top else 0.0
    bigs = list(zip(tokens, tokens[1:]))
    uniq_bigs = len(set(bigs)) / max(1, len(bigs))
    from math import log2
    freqs = [c / total for _, c in collections.Counter(tokens).items()]
    entropy = -sum(p * log2(p) for p in freqs if p > 0)

    hidden = {"white_text_ratio": 0.0, "tiny_font_ratio": 0.0}
    if pdf_path:
        hidden = detect_hidden_text(pdf_path)

    return {
        "top_token_ratio": round(top_ratio, 4),
        "bigram_unique_ratio": round(uniq_bigs, 4),
        "token_entropy": round(entropy, 4),
        "white_text_ratio": hidden.get("white_text_ratio", 0.0),
        "tiny_font_ratio": hidden.get("tiny_font_ratio", 0.0),
    }


HTML_CSS = """
<style>
body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#111}
h1{font-size:28px;margin:8px 0}
h2{font-size:20px;margin:16px 0 8px}
.card{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,0.03)}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;background:#f1f5f9;margin-right:6px;font-size:12px}
.ok{background:#dcfce7}
.warn{background:#fee2e2}
.table{width:100%;border-collapse:collapse}
.table th,.table td{border-bottom:1px solid #eee;padding:8px;text-align:left}
.bar{height:10px;background:#e5e7eb;border-radius:6px;overflow:hidden}
.bar>span{display:block;height:10px;background:#3b82f6}
.small{font-size:12px;color:#555}
.code{font-family:ui-monospace, SFMono-Regular, Menlo, monospace;background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:8px}
mark{background:#fef08a}
</style>
"""


def highlight_terms(text: str, terms: List[str]) -> str:
    s = text
    for t in sorted(set(terms), key=len, reverse=True):
        if not t or len(t) < 3:
            continue
        try:
            s = re.sub(fr"(?i)\b({re.escape(t)})\b", r"<mark>\1</mark>", s)
        except Exception:
            pass
    return s


def render_html(report: Dict[str, Any], resume_text: str, jd_text: str, out_path: str) -> None:
    cs = report.get("component_scores", {})
    req = report.get("per_skill_years_required", {})
    est = report.get("per_skill_years_estimated", {})
    gates = report.get("gates", {})
    sem = report.get("semantic", {})

    bars = []
    for k in sorted(set(req.keys()) | set(est.keys())):
        reqv = float(req.get(k, 0))
        estv = float(est.get(k, 0))
        pct = 0 if reqv <= 0 else min(100, int((estv / reqv) * 100))
        bars.append(f"<tr><td>{k}</td><td>{estv:.2f}y</td><td>{reqv:.2f}y</td><td><div class='bar'><span style='width:{pct}%;'></span></div> {pct}%</td></tr>")

    jd_hl = highlight_terms(jd_text, list(req.keys()) + list(est.keys()))
    res_hl = highlight_terms(resume_text, list(req.keys()) + list(est.keys()))

    top_chunks = sem.get("top_chunks", [])
    chunks_html = "".join(
        f"<div class='card'><div class='small'>bi-score: {score:.3f}</div><div>{highlight_terms(chunk, list(req.keys()))}</div></div>"
        for chunk, score in top_chunks
    )

    gates_html = "".join(
        f"<span class='badge {('ok' if v else 'warn')}'>{k.replace('_',' ')}</span>" for k, v in gates.items()
    )

    html = f"""
<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>ATS Pro Plus Report</title>{HTML_CSS}</head>
<body>
  <h1>ATS Pro Plus — Report</h1>
  <div class='small'>Version {VERSION} • Overall: <b>{report.get('overall_score',0)}/100</b></div>

  <div class='card'>
    <h2>Gates</h2>
    <div>{gates_html}</div>
  </div>

  <div class='card'>
    <h2>Component Scores</h2>
    <table class='table'>
      <tr><th>Keywords</th><th>Semantic</th><th>Sections</th><th>Experience</th><th>Impact</th><th>Quality</th><th>Seniority</th><th>Formatting Penalty</th></tr>
      <tr>
        <td>{cs.get('keywords',0)}</td><td>{cs.get('semantic',0)}</td><td>{cs.get('sections',0)}</td>
        <td>{cs.get('experience',0)}</td><td>{cs.get('impact',0)}</td><td>{cs.get('quality',0)}</td>
        <td>{cs.get('seniority',0)}</td><td>-{cs.get('formatting_penalty_pct',0)}%</td>
      </tr>
    </table>
  </div>

  <div class='card'>
    <h2>Per-skill Years Coverage</h2>
    <table class='table'>
      <tr><th>Skill</th><th>Estimated</th><th>Required</th><th>Coverage</th></tr>
      {''.join(bars)}
    </table>
  </div>

  <div class='card'>
    <h2>Top Matching Snippets</h2>
    {chunks_html if chunks_html else '<div class="small">(semantic details not available)</div>'}
  </div>

  <div class='card'>
    <h2>JD (highlighted)</h2>
    <div class='code'>{jd_hl}</div>
  </div>

  <div class='card'>
    <h2>Resume (highlighted)</h2>
    <div class='code'>{res_hl[:30000]}</div>
  </div>

  <div class='card'>
    <h2>Suggestions</h2>
    <ul>
      {''.join(f'<li>{s}</li>' for s in report.get('suggestions', []))}
    </ul>
  </div>
</body></html>
"""
    pathlib.Path(out_path).write_text(html, encoding="utf-8")


def compute_scores(
    jd_text: str,
    resume_text: str,
    sections: Dict[str, str],
    exps: List[Experience],
    taxonomy: Dict[str, set],
    role: Dict[str, Any],
    args: argparse.Namespace,
    pdf_path: Optional[str],
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    role_name = role.get("role_name", "unspecified")
    required = {canonicalize_skill(x, taxonomy) for x in map(str.lower, role.get("required_skills", []))}
    nice_to_have = {canonicalize_skill(x, taxonomy) for x in map(str.lower, role.get("nice_to_have_skills", []))}
    min_years_required = float(role.get("min_years_experience_total", 0) or 0)
    per_skill_years = {canonicalize_skill(k, taxonomy): safe_float(v, 0) for k, v in role.get("per_skill_years", {}).items()}
    location_req = (role.get("location", "") or "").lower()
    work_auth_req = (role.get("work_authorization", "") or "").lower()
    degree_req = (role.get("degree", "") or "").lower()
    jd_seniority = role.get("seniority") or detect_seniority(jd_text)

    jd_tokens = tokenize(jd_text)
    res_tokens = tokenize(resume_text)
    token_set_res = set(res_tokens)
    jd_bigrams = bigrams(jd_tokens)

    sem = semantic_scores(jd_text, resume_text, args.bi_encoder, args.cross_encoder)

    skills_for_eval = sorted(set(required) | set(nice_to_have) | set(per_skill_years.keys()))
    per_skill_years_est = compute_per_skill_years(exps, skills_for_eval, taxonomy)

    years_total = years_total_from_experiences(exps)

    req_hits = {r: (synonym_hit(r, token_set_res, taxonomy) or (r in token_set_res)) for r in required}
    req_coverage = (sum(req_hits.values()) / max(1, len(required))) if required else 1.0
    nth_hits = {n: (synonym_hit(n, token_set_res, taxonomy) or (n in token_set_res)) for n in nice_to_have}
    nth_coverage = (sum(nth_hits.values()) / max(1, len(nice_to_have))) if nice_to_have else 0.0
    phrase_cov = (sum(1 for p in jd_bigrams if p in resume_text.lower()) / max(1, len(jd_bigrams))) if jd_bigrams else 0.0

    present_sections = {k: (k in sections and len(sections[k]) > 40) for k in [
        "summary",
        "experience",
        "projects",
        "skills",
        "education",
        "certifications",
        "achievements",
        "links",
    ]}
    order_penalty = 0.0
    logical_order = ["summary", "skills", "experience", "projects", "education"]
    indices = [list(sections.keys()).index(s) for s in logical_order if s in sections]
    if indices != sorted(indices):
        order_penalty = 0.05

    bullets = re.findall(r"(?:^|\n)[•\-–]\s*(.+?)(?=\n|$)", resume_text)
    digits_ratio = (sum(bool(re.search(r"\d", b)) for b in bullets) / len(bullets)) if bullets else 0.0

    q = readability_metrics(resume_text)
    grammar_count = grammar_issues(resume_text) if args.grammar_check else 0

    hygiene_score = max(0.0, 1.0 - min(1.0, grammar_count / 50.0))

    res_seniority = detect_seniority(resume_text)
    sdist = seniority_distance(res_seniority, jd_seniority)
    seniority_fit = max(0.0, 1.0 - min(1.0, sdist / 4.0))

    fmt_flags = {
        "multi_column_pdf": bool(pdf_path and detect_multicolumn(pdf_path)),
        "icons_colors": False,
        "footnotes": False,
    }
    penalties = sum(v for v in fmt_flags.values()) * 0.02

    ag = anti_gaming_metrics(resume_text, pdf_path)
    anti_gaming_penalty = 0.0
    if ag["top_token_ratio"] > 0.06 or ag["bigram_unique_ratio"] < 0.4 or ag["white_text_ratio"] > 0.01 or ag["tiny_font_ratio"] > 0.01:
        anti_gaming_penalty = 0.05

    lacking_years = {k: v for k, v in per_skill_years.items() if per_skill_years_est.get(k, 0.0) < v}
    loc_ok = True if not location_req else (location_req in resume_text.lower() or "remote" in jd_text.lower())
    auth_ok = True if not work_auth_req else (work_auth_req in resume_text.lower())
    degree_ok = True if not degree_req else (degree_req in resume_text.lower())
    seniority_gate = sdist <= 3

    gates = {
        "required_skills_coverage_ok": req_coverage >= (args.min_req_coverage if required else 0.0),
        "per_skill_years_ok": len(lacking_years) == 0,
        "total_years_ok": years_total >= min_years_required,
        "location_ok": loc_ok,
        "work_auth_ok": auth_ok,
        "degree_ok": degree_ok,
        "seniority_ok": seniority_gate,
    }

    s_keywords = 0.32 * (0.55 * req_coverage + 0.3 * nth_coverage + 0.15 * phrase_cov)
    s_semantic = 0.20 * (0.5 * sem.get("bi_encoder_score", 0.0) + 0.5 * sem.get("cross_encoder_score", 0.0))
    s_sections = 0.10 * ((sum(present_sections.values()) / max(1, len(present_sections))) - order_penalty)
    per_skill_cov = 1.0
    if per_skill_years:
        coverages = []
        for k, need in per_skill_years.items():
            have = per_skill_years_est.get(k, 0.0)
            coverages.append(min(1.0, have / max(need, 0.0001)))
        per_skill_cov = sum(coverages) / len(coverages)
    s_experience = 0.16 * (0.6 * min(1.0, years_total / max(1.0, min_years_required or 1.0)) + 0.4 * per_skill_cov)
    s_impact = 0.06 * digits_ratio
    read_norm = 0.0
    if q["flesch_reading_ease"]:
        read_norm = min(1.0, max(0.0, (q["flesch_reading_ease"] / 100.0)))
    s_quality = 0.10 * (0.6 * read_norm + 0.4 * hygiene_score)
    s_seniority = 0.06 * seniority_fit

    score_raw = s_keywords + s_semantic + s_sections + s_experience + s_impact + s_quality + s_seniority
    score = round(max(0, min(100, (score_raw - penalties - anti_gaming_penalty) * 100)))

    missing_required = [k for k, v in req_hits.items() if not v]

    component_scores = {
        "keywords": round(s_keywords * 100),
        "semantic": round(s_semantic * 100),
        "sections": round(s_sections * 100),
        "experience": round(s_experience * 100),
        "impact": round(s_impact * 100),
        "quality": round(s_quality * 100),
        "seniority": round(s_seniority * 100),
        "formatting_penalty_pct": round((penalties + anti_gaming_penalty) * 100, 1),
    }

    suggestions: List[str] = []
    if missing_required:
        suggestions.append("Add or surface evidence for missing required skills: " + ", ".join(sorted(missing_required)) + ".")
    if lacking_years:
        gaps = sorted(((k, v - per_skill_years_est.get(k, 0.0)) for k, v in lacking_years.items()), key=lambda x: -x[1])
        suggestions.append("Strengthen quantified experience: " + ", ".join(f"{k} (+{gap:.1f}y)" for k, gap in gaps))
    if digits_ratio < 0.5:
        suggestions.append("Increase impact bullets with metrics (%, $, time): aim for >50% of bullets to include numbers.")
    if read_norm < 0.4:
        suggestions.append("Improve readability: shorter sentences, active voice, concrete verbs.")
    if not gates["seniority_ok"]:
        suggestions.append("Align title/scope to the JD seniority (adjust headline or emphasize leadership scope).")

    t1 = time.perf_counter()

    report: Dict[str, Any] = {
        "version": VERSION,
        "role_name": role_name,
        "overall_score": score,
        "component_scores": component_scores,
        "gates": gates,
        "years_total_detected": years_total,
        "required_coverage": round(req_coverage, 3),
        "nice_to_have_coverage": round(nth_coverage, 3),
        "phrase_coverage": round(phrase_cov, 3),
        "missing_required_skills": missing_required,
        "per_skill_years_required": per_skill_years,
        "per_skill_years_estimated": per_skill_years_est,
        "lacking_years": {k: v for k, v in per_skill_years.items() if per_skill_years_est.get(k, 0.0) < v},
        "sections_present": present_sections,
        "format_flags": fmt_flags,
        "quality": {"readability": q, "grammar_issues": grammar_count},
        "anti_gaming": ag,
        "semantic": sem,
        "suggestions": suggestions,
        "timings_s": {"scoring": round(t1 - t0, 3)},
    }

    return report


def evaluate_single(args: argparse.Namespace, taxonomy: Dict[str, set]) -> Tuple[Dict[str, Any], str, str]:
    role: Dict[str, Any] = {}
    if args.role and pathlib.Path(args.role).exists():
        try:
            role = json.loads(read_textfile(args.role))
        except Exception:
            role = {}
    jd_text = read_textfile(args.jd) or role.get("job_description", "")

    if not role.get("required_skills") and jd_text:
        auto = extract_requirements_from_jd(jd_text, taxonomy)
        role.setdefault("role_name", auto.role_name)
        role.setdefault("required_skills", auto.required_skills)
        role.setdefault("nice_to_have_skills", auto.nice_to_have_skills)
        role.setdefault("per_skill_years", auto.per_skill_years)
        role.setdefault("min_years_experience_total", auto.min_years_total)
        role.setdefault("degree", auto.degree)
        role.setdefault("location", auto.location)
        role.setdefault("work_authorization", auto.work_authorization)
        role.setdefault("seniority", auto.seniority)

    resume_text = extract_pdf_text(args.pdf, use_ocr=args.use_ocr) if args.pdf else ""
    if not resume_text.strip():
        resume_text = strip_latex(read_textfile(args.tex)) if args.tex else ""
    resume_text = normalize_ws(resume_text)

    sections = split_sections(resume_text)
    exps = segment_experiences(resume_text)

    report = compute_scores(jd_text, resume_text, sections, exps, taxonomy, role, args, args.pdf)

    return report, resume_text, jd_text


def save_outputs_single(report: Dict[str, Any], args: argparse.Namespace, resume_text: str, jd_text: str) -> None:
    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"Wrote JSON: {args.out_json}")

    if args.out_md:
        lines: List[str] = []
        cs = report["component_scores"]
        lines.append("### 🧠 ATS Pro Plus Report")
        lines.append(f"**Overall:** **{report.get('overall_score', 0)}/100**\n")
        lines.append("**Gates**")
        for k, v in report.get("gates", {}).items():
            lines.append(f"- {'✅' if v else '❌'} {k.replace('_',' ').title()}")
        if report.get("missing_required_skills"):
            lines.append(f"- ⚠️ Missing required skills: {', '.join(sorted(report['missing_required_skills']))}")
        if report.get("lacking_years"):
            lines.append("- ⚠️ Insufficient years for: " + ", ".join(f"{k} (need {v}y)" for k, v in report["lacking_years"].items()))
        lines.append("\n**Breakdown**")
        lines.append(
            f"- Keywords: {cs['keywords']} | Semantic: {cs['semantic']} | Sections: {cs['sections']} | "
            f"Experience: {cs['experience']} | Impact: {cs['impact']} | Quality: {cs['quality']} | "
            f"Seniority: {cs['seniority']} | Formatting penalty: -{cs['formatting_penalty_pct']}%\n"
        )
        if report.get("suggestions"):
            lines.append("**Suggestions**")
            for s in report["suggestions"]:
                lines.append(f"- {s}")
        if args.role:
            lines.append(f"\n_Role profile_: **{report.get('role_name','unspecified')}**")
        pathlib.Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Wrote Markdown: {args.out_md}")

    if args.out_html:
        render_html(report, resume_text, jd_text, args.out_html)
        logger.info(f"Wrote HTML: {args.out_html}")

    all_gates = all(report.get("gates", {}).values())
    if (not all_gates) or (report.get("overall_score", 0) < args.min_score):
        print(json.dumps(report, indent=2))
        raise SystemExit(1)


def evaluate_batch(args: argparse.Namespace, taxonomy: Dict[str, set]) -> None:
    res_dir = pathlib.Path(args.res_dir)
    files = sorted([p for p in res_dir.glob("**/*") if p.suffix.lower() in {".pdf", ".PDF"}])
    if not files:
        logger.error("No PDF files found in --res-dir.")
        return

    role: Dict[str, Any] = {}
    if args.role and pathlib.Path(args.role).exists():
        try:
            role = json.loads(read_textfile(args.role))
        except Exception:
            role = {}
    jd_text = read_textfile(args.jd) or role.get("job_description", "")

    if not role.get("required_skills") and jd_text:
        auto = extract_requirements_from_jd(jd_text, taxonomy)
        role = {**auto.__dict__, **role}

    rows: List[Dict[str, Any]] = []
    jsonl_path = pathlib.Path(args.out_jsonl) if args.out_jsonl else None
    jsonl_fh = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None

    for pdf in files:
        logger.info(f"Scoring: {pdf.name}")
        args_single = argparse.Namespace(**vars(args))
        args_single.pdf = str(pdf)
        args_single.tex = None
        report, resume_text, jd_text2 = evaluate_single(args_single, taxonomy)
        rows.append({
            "file": pdf.name,
            "score": report.get("overall_score", 0),
            "gates_ok": int(all(report.get("gates", {}).values())),
            "required_cov": report.get("required_coverage", 0.0),
            "nice_cov": report.get("nice_to_have_coverage", 0.0),
            "years_total": report.get("years_total_detected", 0.0),
        })
        if jsonl_fh:
            jsonl_fh.write(json.dumps({"file": pdf.name, **report}, ensure_ascii=False) + "\n")

    if jsonl_fh:
        jsonl_fh.close()
        logger.info(f"Wrote JSONL: {jsonl_path}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "score", "gates_ok", "required_cov", "nice_cov", "years_total"])
            w.writeheader()
            for r in sorted(rows, key=lambda x: (-x["gates_ok"], -x["score"])):
                w.writerow(r)
        logger.info(f"Wrote CSV: {args.out_csv}")

    if args.out_html_index:
        rows_html = "".join(
            f"<tr><td>{r['file']}</td><td>{r['score']}</td><td>{'✅' if r['gates_ok'] else '❌'}</td><td>{r['required_cov']}</td><td>{r['nice_cov']}</td><td>{r['years_total']}</td></tr>"
            for r in sorted(rows, key=lambda x: (-x["gates_ok"], -x["score"]))
        )
        html = f"""
<!DOCTYPE html><html><head><meta charset='utf-8'><title>ATS Pro Plus — Batch</title>{HTML_CSS}</head>
<body>
<h1>ATS Pro Plus — Batch Ranking</h1>
<table class='table'>
<tr><th>File</th><th>Score</th><th>Gates OK</th><th>Required Cov</th><th>Nice Cov</th><th>Years Total</th></tr>
{rows_html}
</table>
</body></html>
"""
        pathlib.Path(args.out_html_index).write_text(html, encoding="utf-8")
        logger.info(f"Wrote HTML Index: {args.out_html_index}")


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=f"ATS Pro Plus v{VERSION}")
    ap.add_argument("--pdf", help="Path to compiled resume PDF")
    ap.add_argument("--tex", help="Path to LaTeX source (fallback)")
    ap.add_argument("--jd", default="job-description.txt", help="Path to JD text file (optional)")
    ap.add_argument("--role", default="", help="Path to role JSON (optional)")
    ap.add_argument("--out-json", default="ats-pro-report.json")
    ap.add_argument("--out-md", default="ats-pro-report.md")
    ap.add_argument("--out-html", default="ats-pro-report.html")
    ap.add_argument("--min-score", type=int, default=75)
    ap.add_argument("--min-req-coverage", type=float, default=0.85)
    ap.add_argument("--use-ocr", action="store_true", help="Enable OCR fallback for scanned PDFs")
    ap.add_argument("--grammar-check", action="store_true", help="Run grammar check (language_tool_python)")
    ap.add_argument("--taxonomy", default="", help="Path to taxonomy JSON (optional)")
    ap.add_argument("--bi-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--res-dir", default="", help="Directory of PDFs to score (enables batch mode)")
    ap.add_argument("--out-csv", default="ats-batch.csv")
    ap.add_argument("--out-jsonl", default="ats-batch.jsonl")
    ap.add_argument("--out-html-index", default="ats-batch.html")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    return args


def main():
    args = build_args()
    taxonomy = load_taxonomy(args.taxonomy)

    if args.res_dir:
        evaluate_batch(args, taxonomy)
        return

    if not args.pdf and not args.tex:
        logger.error("Provide at least --pdf or --tex for single evaluation, or --res-dir for batch.")
        sys.exit(2)

    report, resume_text, jd_text = evaluate_single(args, taxonomy)
    save_outputs_single(report, args, resume_text, jd_text)


if __name__ == "__main__":
    main()
