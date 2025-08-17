#!/usr/bin/env python3
"""
ATS Pro Plus — A production-grade, explainable ATS scoring script
-----------------------------------------------------------------
Persona: Senior Software Architect & Senior Software Developer (20+ yrs)

Key upgrades in v2.1.0:
  ✓ External taxonomy JSON with categories (hard/soft/methodologies/domains/certs/titles/seniority/scope)
  ✓ Optional JD-filtered taxonomy: use only skills that appear in the JD (but keep synonyms for resume matching)
  ✓ Scoring dimensions expanded:
      - Hard-skill match (JD-driven, canonical+synonyms, family expansion)
      - Years per skill (timeline-scoped by experience blocks)
      - Recency of relevant work (months since last use per required skill)
      - Title & seniority alignment (resume vs JD)
      - Role scope fit (IC vs lead/manager)
      - Domain/industry relevance
      - Methodologies & architecture coverage (microservices, CI/CD, testing, etc.)
      - Breadth vs depth (unique skills matched vs deeper experience)
      - Project impact & metrics (bullets with numbers / outcomes verbs)
      - Keyword placement (Summary + Experience vs only Skills list)
      - Career progression & stability (promotion signals, short stints)
      - Education & certifications (degree presence & mapped certs)
      - JD coverage score (must-haves vs nice-to-haves)
      - Readability & clarity (textstat) + optional grammar
      - Format & parse-ability (multi-column, OCR fallback, hidden text)
      - Contact & links extraction (email/phone/LinkedIn/GitHub)
      - Chronology consistency (gaps/overlaps)
      - Anti-gaming checks (stuffing, white text, tiny fonts)
      - Compliance/constraints (location/auth/onsite-ready)
  ✓ Scoring-config JSON to tweak weights & thresholds without code changes
  ✓ Batch ranking + HTML reports, JSON/MD, observability & timings

Notes:
- Optional deps are used if present and gracefully degraded if missing.
- No external network calls — models must be installed locally if used.
- Single-file design for drop-in use.

Usage examples:
  # Single resume
  python ats_pro_plus.py \
    --pdf resume.pdf \
    --tex resume.tex \
    --jd job-description.txt \
    --taxonomy taxonomy.json \
    --scoring-config scoring.json \
    --jd-skill-filter \
    --out-json report.json \
    --out-md report.md \
    --out-html report.html \
    --min-score 75

  # Batch mode (directory of PDFs)
  python ats_pro_plus.py \
    --res-dir ./resumes \
    --jd job-description.txt \
    --taxonomy taxonomy.json \
    --scoring-config scoring.json \
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

# ---------- Optional deps ----------
try:
    import dateparser
except Exception:
    dateparser = None

try:
    from rapidfuzz import fuzz
except Exception:
    class _F:
        @staticmethod
        def partial_ratio(a, b): return 100 if a == b else 0
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
    import fitz  # PyMuPDF
except Exception:
    fitz = None

VERSION = "2.1.0"

logger = logging.getLogger("ATSProPlus")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------- Utils ----------
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

# ---------- PDF/LaTeX ----------
def extract_pdf_text(pdf_path: str, use_ocr: bool = False) -> str:
    text = ""
    if fitz is not None:
        try:
            with fitz.open(pdf_path) as doc:
                parts = [page.get_text("text") for page in doc]
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

def normalize_bullets(s: str) -> str:
    # put a newline before any " • " that isn't already at start of a line
    s = re.sub(r"(?<!^)\s*•\s+", r"\n• ", s)
    # also catch dash bullets that appear mid-line: " - " or " – "
    s = re.sub(r"(?<!^)\s+[–\-]\s+(?=[A-Z0-9])", r"\n- ", s)
    return s
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

# ---------- Tokenization ----------
def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+.#-]{2,}", (s or "").lower())

def bigrams(tokens: List[str]) -> set:
    return {" ".join(p) for p in zip(tokens, tokens[1:])}

# ---------- Taxonomy bundle ----------
@dataclass
class TaxonomyBundle:
    canon: Dict[str, set]                      # canonical -> synonyms
    hard: set                                  # canonical hard skills
    soft: set                                  # canonical soft skills
    methodologies: set                         # canonical methodologies/architecture
    domains: set                               # canonical domains/industries
    families: Dict[str, set]                   # family -> canonical skills
    certs: Dict[str, set]                      # canonical skill -> cert phrases
    titles_ic: set                             # IC titles (engineer, developer...)
    titles_lead: set                           # lead/manager/architect/head
    seniority_terms: Dict[str, set]            # seniority level -> phrases
    role_scope_terms: Dict[str, set]           # "ic"/"lead"/"manager" -> phrases

def _as_set(value) -> set:
    if value is None: return set()
    if isinstance(value, list): return set(map(str.lower, value))
    if isinstance(value, set): return set(map(str.lower, value))
    if isinstance(value, dict): return set(map(str.lower, value.keys()))
    return set([str(value).lower()])

def load_taxonomy(path: Optional[str]) -> TaxonomyBundle:
    """
    Expected JSON schema (example at bottom of file):
    {
      "schema_version": "1.0",
      "skills": {
        "hard": { "java": ["spring","spring boot"], "kubernetes": ["k8s"], ... },
        "soft": { "communication": ["communicator","stakeholder communication"], ... },
        "methodologies": { "microservices": ["service oriented","soa"], "ci/cd":["ci cd",...], ... },
        "domains": { "fintech": ["payments","banking"], "adtech":["advertising", "rtb"], ... }
      },
      "families": { "cloud": ["aws","gcp","azure"], "frontend":["react","angular","vue"] },
      "certifications": { "aws": ["aws certified", "aws saa", ...], "gcp": ["professional cloud ..."] },
      "searchables": { "rest": ["restful"], "graphql":[] },
      "titles": { "ic": ["engineer","developer","sde","contributor"], "lead": ["lead","manager","architect","head"] },
      "seniority": { "junior":["jr","junior","i"], "mid":["ii","mid"], "senior":["sr","senior","iii"], "staff":["staff"], "principal":["principal"], "architect":["architect"] },
      "role_scope": { "ic":["individual contributor","ic"], "lead":["lead","leading"], "manager":["manager","managing"] }
    }
    """
    if not path or not pathlib.Path(path).exists():
        logger.warning("No taxonomy file provided or not found. Using minimal empty taxonomy.")
        return TaxonomyBundle(
            canon={}, hard=set(), soft=set(), methodologies=set(), domains=set(),
            families={}, certs={}, titles_ic=set(), titles_lead=set(),
            seniority_terms={}, role_scope_terms={}
        )
    try:
        data = json.loads(read_textfile(path))
    except Exception:
        logger.error("Failed to parse taxonomy JSON; using empty taxonomy.")
        data = {}

    skills = data.get("skills", {})
    hard = {k.lower(): set(map(str.lower, v or [])) for k, v in (skills.get("hard") or {}).items()}
    soft = {k.lower(): set(map(str.lower, v or [])) for k, v in (skills.get("soft") or {}).items()}
    methodologies = {k.lower(): set(map(str.lower, v or [])) for k, v in (skills.get("methodologies") or {}).items()}
    domains = {k.lower(): set(map(str.lower, v or [])) for k, v in (skills.get("domains") or {}).items()}
    searchables = {k.lower(): set(map(str.lower, v or [])) for k, v in (data.get("searchables") or {}).items()}

    # Flatten everything into a canonical -> synonyms map
    canon: Dict[str, set] = {}
    def _ingest(block: Dict[str, set]):
        for k, syns in block.items():
            canon.setdefault(k, set()).update(set(syns))
    for block in (hard, soft, methodologies, domains, searchables):
        _ingest(block)

    families = {k.lower(): set(map(str.lower, v or [])) for k, v in (data.get("families") or {}).items()}
    certs = {k.lower(): set(map(str.lower, v or [])) for k, v in (data.get("certifications") or {}).items()}

    titles = data.get("titles", {})
    titles_ic = _as_set(titles.get("ic"))
    titles_lead = _as_set(titles.get("lead"))

    seniority_terms = {k.lower(): _as_set(v) for k, v in (data.get("seniority") or {}).items()}
    role_scope_terms = {k.lower(): _as_set(v) for k, v in (data.get("role_scope") or {}).items()}

    return TaxonomyBundle(
        canon=canon,
        hard=set(hard.keys()),
        soft=set(soft.keys()),
        methodologies=set(methodologies.keys()),
        domains=set(domains.keys()),
        families=families,
        certs=certs,
        titles_ic=titles_ic,
        titles_lead=titles_lead,
        seniority_terms=seniority_terms,
        role_scope_terms=role_scope_terms
    )

# JD-filtered taxonomy subset
def subset_taxonomy_for_jd(jd_text: str, tax: TaxonomyBundle) -> TaxonomyBundle:
    tl = (jd_text or "").lower()
    toks = set(tokenize(tl))

    def _appear(canon_key: str, syns: set) -> bool:
        names = {canon_key} | set(syns)
        for term in names:
            if " " in term:
                if re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", tl):
                    return True
            else:
                if term in toks:
                    return True
        return False

    active_canon: Dict[str, set] = {}
    # Filter across canon keys
    for k, syns in tax.canon.items():
        if _appear(k, syns):
            active_canon[k] = set(syns)

    def _filter_keys(keys: set) -> set:
        return set(k for k in keys if k in active_canon)

    return TaxonomyBundle(
        canon=active_canon,
        hard=_filter_keys(tax.hard),
        soft=_filter_keys(tax.soft),
        methodologies=_filter_keys(tax.methodologies),
        domains=_filter_keys(tax.domains),
        families={fam: set(v for v in vs if v in active_canon) for fam, vs in tax.families.items()},
        certs={k: v for k, v in tax.certs.items() if k in active_canon},
        titles_ic=tax.titles_ic,               # titles/scope/seniority stay global
        titles_lead=tax.titles_lead,
        seniority_terms=tax.seniority_terms,
        role_scope_terms=tax.role_scope_terms
    )

# ---------- Canonicalization ----------
def canonicalize_skill(term: str, canon_map: Dict[str, set]) -> str:
    t = (term or "").lower().strip()
    if t in canon_map:
        return t
    for k, syns in canon_map.items():
        if t == k or t in syns:
            return k
    # fuzzy
    for k, syns in canon_map.items():
        try:
            if fuzz.partial_ratio(t, k) >= 92:
                return k
            if any(fuzz.partial_ratio(t, s) >= 92 for s in syns):
                return k
        except Exception:
            pass
    return t

def synonym_hit(term: str, tokenset: set, canon_map: Dict[str, set]) -> bool:
    t = term.lower()
    if t in tokenset:
        return True
    canonical = canonicalize_skill(t, canon_map)
    if canonical in tokenset:
        return True
    syns = canon_map.get(canonical, set())
    return any(any(fuzz.partial_ratio(syn, tok) >= 90 for tok in tokenset) for syn in syns)

# ---------- Sections & Dates ----------
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
    r"(?P<a>(?:[A-Za-z]{3,9}\s+\d{4}|\d{1,2}/\d{4}|\d{4}))\s*(?:–|—|-|to|till|until)\s*"
    r"(?P<b>(?:Present|Current|Now|\d{4}|[A-Za-z]{3,9}\s+\d{4}|\d{1,2}/\d{4}))",
    re.I,
)

def normalize_dates_text(s: str) -> str:
    s = re.sub(r"\b(till|until)\b", "to", s, flags=re.I)
    s = re.sub(r"\b(current|now)\b", "Present", s, flags=re.I)
    s = s.replace("—", "–").replace("-", "–")  # unify dashes to EN DASH
    return s

def split_blocks_on_multi_ranges(blocks: list[str]) -> list[str]:
    new_blocks = []
    for blk in blocks:
        matches = list(DATE_RANGE_RE.finditer(blk))
        if len(matches) <= 1:
            new_blocks.append(blk)
            continue
        # cut the block at each subsequent date range start
        starts = [m.start() for m in matches]
        for i, (a, b) in enumerate(zip(starts, starts[1:] + [len(blk)])):
            seg = blk[a:b].strip()
            # include the heading portion before first date
            if i == 0:
                head = blk[:a].strip()
                if head:
                    seg = (head + "\n" + seg).strip()
            if seg:
                new_blocks.append(seg)
    return new_blocks

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

# ---------- Experience segmentation ----------
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

def extract_location_req(jd_text: str) -> str:
    # super-light heuristic: look for a capitalized city/region or 'onsite'
    if re.search(r"\b(onsite|on-site)\b", jd_text, re.I):
        return "onsite"
    # If it only says remote/hybrid/flexible, don't treat as a hard location
    if re.search(r"\b(remote|hybrid|flexible)\b", jd_text, re.I):
        return ""
    # otherwise try to capture a location-like token (very naive)
    m = re.search(r"\b(?:Bangalore|Bengaluru|Pune|Hyderabad|India|USA|UK|Singapore|Canada)\b", jd_text)
    return m.group(0).lower() if m else ""
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

    blocks = split_blocks_on_multi_ranges(blocks)


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

def compute_per_skill_years(exps: List[Experience], skills: List[str], canon_map: Dict[str, set]) -> Dict[str, float]:
    res: Dict[str, float] = {canonicalize_skill(s, canon_map): 0.0 for s in skills}
    for e in exps:
        role_text = "\n".join([e.title, e.company] + e.bullets).lower()
        token_set = set(tokenize(role_text))
        for s in list(res.keys()):
            if synonym_hit(s, token_set, canon_map):
                res[s] += e.months
    return {k: round(v / 12.0, 2) for k, v in res.items()}

def compute_last_used_months(exps: List[Experience], skills: List[str], canon_map: Dict[str, set]) -> Dict[str, Optional[int]]:
    """Return months since last use for each skill; None if never used."""
    now = datetime.now()
    out: Dict[str, Optional[int]] = {}
    for s in skills:
        s_c = canonicalize_skill(s, canon_map)
        last: Optional[datetime] = None
        for e in exps:
            role_text = "\n".join([e.title, e.company] + e.bullets).lower()
            if synonym_hit(s_c, set(tokenize(role_text)), canon_map):
                end = e.end or now
                if (last is None) or (end > last):
                    last = end
        if last is None:
            out[s_c] = None
        else:
            months = (now.year - last.year) * 12 + (now.month - last.month)
            out[s_c] = max(0, months)
    return out

# ---------- Seniority / Scope ----------
SENIORITY_ORDER = ["intern", "junior", "associate", "mid", "senior", "lead", "staff", "principal", "architect", "vp", "head"]

def detect_seniority(text: str, seniority_terms: Optional[Dict[str, set]] = None) -> Optional[str]:
    t = (text or "").lower()
    # priority by explicit term map if provided
    if seniority_terms:
        for level, terms in seniority_terms.items():
            if any(re.search(rf"\b{re.escape(term)}\b", t) for term in terms | {level}):
                return level
    # fallback heuristic
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

def detect_role_scope(text: str, role_scope_terms: Optional[Dict[str, set]] = None) -> str:
    t = (text or "").lower()
    if role_scope_terms:
        for scope, terms in role_scope_terms.items():
            if any(re.search(rf"\b{re.escape(term)}\b", t) for term in terms | {scope}):
                return scope
    # heuristic
    if re.search(r"\b(manager|managing|managed|team lead|led|leadership|architect)\b", t):
        return "lead"
    return "ic"

# ---------- JD extraction ----------
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
    required_skills: List[str]             # hard skills inferred
    nice_to_have_skills: List[str]
    per_skill_years: Dict[str, float]
    min_years_total: float
    degree: str
    location: str
    work_authorization: str
    seniority: Optional[str]
    soft_required: List[str]
    domains: List[str]
    methodologies: List[str]
    role_scope: Optional[str]

def extract_requirements_from_jd(jd_text: str, tax: TaxonomyBundle) -> AutoJD:
    t = jd_text or ""
    tl = t.lower()
    tokens = set(tokenize(tl))

    def _hits_from(keys: set) -> set:
        found = set()
        for base in keys:
            syns = tax.canon.get(base, set())
            if (base in tokens) or (" " in base and base in tl) or any((s in tokens) or (" " in s and s in tl) for s in syns):
                found.add(base)
        return found

    hard_found = _hits_from(tax.hard)
    soft_found = _hits_from(tax.soft)
    domains_found = _hits_from(tax.domains)
    methodologies_found = _hits_from(tax.methodologies)

    per_skill: Dict[str, float] = {}
    for s in hard_found:
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

    sr = detect_seniority(tl, tax.seniority_terms)
    scope = detect_role_scope(tl, tax.role_scope_terms)

    role_name = "unspecified"
    m_role = re.search(r"^\s*(?:role|position|title)\s*:\s*(.+)$", tl, flags=re.M)
    if m_role:
        role_name = m_role.group(1)[:80]

    # Nice-to-have heuristic: anything in canon that appears but not in hard_found
    nice: List[str] = sorted(list((_hits_from(set(tax.canon.keys())) - hard_found) - soft_found))

    return AutoJD(
        role_name=role_name,
        required_skills=sorted(hard_found),
        nice_to_have_skills=nice,
        per_skill_years=per_skill,
        min_years_total=my,
        degree=deg,
        location=loc,
        work_authorization=wa,
        seniority=sr,
        soft_required=sorted(soft_found),
        domains=sorted(domains_found),
        methodologies=sorted(methodologies_found),
        role_scope=scope,
    )

# ---------- Semantics ----------
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

# ---------- Quality / Hygiene ----------
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

# ---------- Contacts / Links ----------
CONTACT_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
CONTACT_PHONE_RE = re.compile(r"(?:\+?\d[\d\-().\s]{7,}\d)")
LINK_URL_RE = re.compile(r"https?://[^\s)]+", re.I)

def contact_links(text: str) -> Dict[str, Any]:
    emails = list(set(CONTACT_EMAIL_RE.findall(text)))
    phones = list(set(CONTACT_PHONE_RE.findall(text)))
    urls = list(set(LINK_URL_RE.findall(text)))
    def _has(domain: str): return any(domain in u.lower() for u in urls)
    links = {
        "emails": emails,
        "phones": phones,
        "linkedin": _has("linkedin.com"),
        "github": _has("github.com"),
        "portfolio": any(not (_has("linkedin.com") or _has("github.com")) for _ in urls) and len(urls) > 0,
        "urls": urls[:20]
    }
    return links

# ---------- HTML ----------
HTML_CSS = """
<style>
body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1080px;margin:24px auto;padding:0 16px;color:#111}
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
.kv{display:flex;flex-wrap:wrap;gap:10px}
.kv .item{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:6px 10px;font-size:12px}
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
    gates = report.get("gates", {})
    sem = report.get("semantic", {})
    per_skill_required = report.get("per_skill_years_required", {})
    per_skill_est = report.get("per_skill_years_estimated", {})
    per_skill_last = report.get("per_skill_last_used_months", {})

    bars = []
    for k in sorted(set(per_skill_required.keys()) | set(per_skill_est.keys())):
        need = float(per_skill_required.get(k, 0))
        have = float(per_skill_est.get(k, 0))
        pct = 0 if need <= 0 else min(100, int((have / need) * 100))
        rec = per_skill_last.get(k)
        rec_s = (f"{rec} mo ago" if isinstance(rec, int) else "n/a")
        bars.append(
            f"<tr><td>{k}</td><td>{have:.2f}y</td><td>{need:.2f}y</td>"
            f"<td><div class='bar'><span style='width:{pct}%;'></span></div> {pct}%</td>"
            f"<td>{rec_s}</td></tr>"
        )

    top_chunks = sem.get("top_chunks", [])
    chunks_html = "".join(
        f"<div class='card'><div class='small'>bi-score: {score:.3f}</div><div>{highlight_terms(chunk, list(per_skill_required.keys()))}</div></div>"
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
    <div class='small'>Min req coverage: {report.get('min_req_coverage',0)} • Total years: {report.get('years_total_detected',0)}</div>
  </div>

  <div class='card'>
    <h2>Component Scores</h2>
    <table class='table'>
      <tr><th>Hard skills</th><th>Soft skills</th><th>Semantic</th><th>Experience</th><th>Per-skill Years</th><th>Recency</th><th>Methodologies</th><th>Domain</th><th>Breadth/Depth</th><th>Impact</th><th>Placement</th><th>Quality</th><th>Seniority</th><th>Role Scope</th><th>Stability</th><th>Compliance</th><th>Formatting Penalty</th></tr>
      <tr>
        <td>{cs.get('hard_keywords',0)}</td><td>{cs.get('soft_keywords',0)}</td><td>{cs.get('semantic',0)}</td>
        <td>{cs.get('experience_total',0)}</td><td>{cs.get('per_skill_years',0)}</td><td>{cs.get('recency',0)}</td>
        <td>{cs.get('methodologies',0)}</td><td>{cs.get('domain',0)}</td><td>{cs.get('breadth_depth',0)}</td>
        <td>{cs.get('impact',0)}</td><td>{cs.get('placement',0)}</td><td>{cs.get('quality',0)}</td>
        <td>{cs.get('seniority',0)}</td><td>{cs.get('role_scope',0)}</td><td>{cs.get('stability',0)}</td>
        <td>{cs.get('compliance',0)}</td><td>-{cs.get('formatting_penalty_pct',0)}%</td>
      </tr>
    </table>
  </div>

  <div class='card'>
    <h2>Per-skill Years & Recency</h2>
    <table class='table'>
      <tr><th>Skill</th><th>Estimated</th><th>Required</th><th>Coverage</th><th>Last used</th></tr>
      {''.join(bars)}
    </table>
  </div>

  <div class='card'>
    <h2>Top Matching Snippets</h2>
    {chunks_html if chunks_html else '<div class="small">(semantic details not available)</div>'}
  </div>

  <div class='card'>
    <h2>JD (highlighted)</h2>
    <div class='code'>{highlight_terms(jd_text, list(per_skill_required.keys()))}</div>
  </div>

  <div class='card'>
    <h2>Resume (highlighted)</h2>
    <div class='code'>{highlight_terms(resume_text[:30000], list(per_skill_required.keys()))}</div>
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

# ---------- Scoring ----------
DEFAULT_WEIGHTS = {
    "hard_keywords": 0.18,
    "soft_keywords": 0.04,
    "semantic": 0.14,
    "experience_total": 0.06,
    "per_skill_years": 0.10,
    "recency": 0.06,
    "methodologies": 0.05,
    "domain": 0.05,
    "breadth_depth": 0.06,
    "impact": 0.05,
    "placement": 0.04,
    "quality": 0.04,
    "seniority": 0.04,
    "role_scope": 0.03,
    "stability": 0.03,
    "compliance": 0.03,
    # formatting/anti-gaming treated as penalty below
}

DEFAULT_THRESHOLDS = {
    "job_hop_months": 9,
    "gap_months_warn": 6,
    "recency_half_life_months": 24,  # for recency decay
}

def load_scoring_config(path: Optional[str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if not path or not pathlib.Path(path).exists():
        return dict(DEFAULT_WEIGHTS), dict(DEFAULT_THRESHOLDS)
    try:
        cfg = json.loads(read_textfile(path))
    except Exception:
        logger.warning("Failed to parse scoring-config JSON. Using defaults.")
        return dict(DEFAULT_WEIGHTS), dict(DEFAULT_THRESHOLDS)
    weights = dict(DEFAULT_WEIGHTS)
    weights.update(cfg.get("weights", {}))
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(cfg.get("thresholds", {}))
    return weights, thresholds

def compute_breadth_depth(required: List[str], per_skill_years_est: Dict[str, float]) -> Tuple[float, float]:
    if not required:
        return 0.0, 0.0
    matched = [s for s in required if per_skill_years_est.get(s, 0.0) > 0]
    breadth = len(matched) / len(required)
    depth = statistics.median([per_skill_years_est.get(s, 0.0) for s in matched]) if matched else 0.0
    # normalize depth (e.g., 0-5y mapped to 0..1)
    depth_norm = min(1.0, depth / 5.0)
    return breadth, depth_norm

def keyword_placement_score(sections: Dict[str, str], skills: List[str], canon_map: Dict[str, set]) -> float:
    if not skills:
        return 0.0
    sum_text = (sections.get("summary") or "").lower()
    exp_text = (sections.get("experience") or "").lower()
    skills_text = (sections.get("skills") or "").lower()
    tokens_sum = set(tokenize(sum_text))
    tokens_exp = set(tokenize(exp_text))
    tokens_sk = set(tokenize(skills_text))
    hits_sum = sum(1 for s in skills if synonym_hit(s, tokens_sum, canon_map))
    hits_exp = sum(1 for s in skills if synonym_hit(s, tokens_exp, canon_map))
    hits_sk = sum(1 for s in skills if synonym_hit(s, tokens_sk, canon_map))
    # Reward presence in Summary and Experience; small credit for Skills list
    denom = max(1, len(skills))
    return min(1.0, (0.4 * hits_sum + 0.5 * hits_exp + 0.1 * hits_sk) / denom)

def stability_progression_score(exps: List[Experience]) -> Tuple[float, Dict[str, Any]]:
    if not exps:
        return 0.0, {"avg_months": 0, "short_roles": 0, "promotions": 0}
    avg_months = statistics.mean([e.months for e in exps]) if exps else 0.0
    short_roles = sum(1 for e in exps if e.months < 9.0)
    # promotions: detect increasing seniority across chronology
    ordered = sorted(exps, key=lambda e: (e.start or datetime(1900,1,1)))
    levels = [detect_seniority(e.title) for e in ordered]
    promotions = 0
    for a, b in zip(levels, levels[1:]):
        if a and b:
            da = SENIORITY_ORDER.index(a) if a in SENIORITY_ORDER else 0
            db = SENIORITY_ORDER.index(b) if b in SENIORITY_ORDER else 0
            if db > da:
                promotions += 1
    # normalize: avg tenure 12-36 months -> 0..1, penalize many short roles
    avg_norm = max(0.0, min(1.0, (avg_months - 12.0) / 24.0))
    penalty = min(1.0, short_roles / max(1, len(exps)))
    score = max(0.0, min(1.0, avg_norm * (1.0 - 0.7 * penalty) + min(1.0, promotions / 3.0) * 0.3))
    return score, {"avg_months": round(avg_months, 1), "short_roles": short_roles, "promotions": promotions}

def chronology_consistency(exps: List[Experience]) -> Dict[str, Any]:
    if not exps:
        return {"gaps_over_6m": 0, "overlaps": 0}
    ordered = sorted(exps, key=lambda e: (e.start or datetime(1900,1,1)))
    gaps = overlaps = 0
    for prev, cur in zip(ordered, ordered[1:]):
        if prev.end and cur.start:
            if cur.start > prev.end:
                months = (cur.start.year - prev.end.year) * 12 + (cur.start.month - prev.end.month)
                if months >= 6: gaps += 1
            if cur.start < (prev.end or cur.start):
                overlaps += 1
    return {"gaps_over_6m": gaps, "overlaps": overlaps}

def edu_certs_score(sections: Dict[str, str], tax: TaxonomyBundle) -> Tuple[float, Dict[str, Any]]:
    edu_text = sections.get("education", "") + " " + sections.get("certifications", "")
    tl = edu_text.lower()
    deg_hit = any(re.search(fr"\b{re.escape(w)}\b", tl) for w in DEGREE_WORDS)
    cert_hits = 0
    for canon, phrases in tax.certs.items():
        for ph in phrases:
            if re.search(fr"(?<![A-Za-z0-9]){re.escape(ph)}(?![A-Za-z0-9])", tl):
                cert_hits += 1
                break
    score = min(1.0, (0.6 if deg_hit else 0.0) + min(0.4, cert_hits * 0.2))
    return score, {"degree": deg_hit, "cert_count": cert_hits}

def compute_scores(
        jd_text: str,
        resume_text: str,
        sections: Dict[str, str],
        exps: List[Experience],
        tax: TaxonomyBundle,
        role: Dict[str, Any],
        args: argparse.Namespace,
        pdf_path: Optional[str],
        weights: Dict[str, float],
        thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    role_name = role.get("role_name", "unspecified")

    # JD-driven requirements
    required_hard = sorted({canonicalize_skill(x, tax.canon) for x in map(str.lower, role.get("required_skills", []))})
    required_soft = sorted({canonicalize_skill(x, tax.canon) for x in map(str.lower, role.get("soft_required", []))})
    nice_to_have = sorted({canonicalize_skill(x, tax.canon) for x in map(str.lower, role.get("nice_to_have_skills", []))})
    per_skill_years_req = {canonicalize_skill(k, tax.canon): safe_float(v, 0) for k, v in role.get("per_skill_years", {}).items()}
    min_years_required = float(role.get("min_years_experience_total", 0) or 0)
    jd_domains = sorted({canonicalize_skill(x, tax.canon) for x in map(str.lower, role.get("domains", []))})
    jd_methodologies = sorted({canonicalize_skill(x, tax.canon) for x in map(str.lower, role.get("methodologies", []))})
    jd_seniority = role.get("seniority") or detect_seniority(jd_text, tax.seniority_terms)
    jd_scope = role.get("role_scope") or detect_role_scope(jd_text, tax.role_scope_terms)
    location_req = (role.get("location", "") or "").lower()
    work_auth_req = (role.get("work_authorization", "") or "").lower()
    degree_req = (role.get("degree", "") or "").lower()

    # Resume tokens
    res_tokens = tokenize(resume_text)
    token_set_res = set(res_tokens)

    # Semantics
    sem = semantic_scores(jd_text, resume_text, args.bi_encoder, args.cross_encoder)

    # Per-skill years + recency
    skills_for_eval = sorted(set(required_hard) | set(per_skill_years_req.keys()))
    per_skill_years_est = compute_per_skill_years(exps, skills_for_eval, tax.canon)
    per_skill_last_used = compute_last_used_months(exps, skills_for_eval, tax.canon)

    years_total = years_total_from_experiences(exps)

    # Hard/soft/NTF coverage
    def _cov(skills: List[str]) -> float:
        if not skills: return 1.0
        return sum((synonym_hit(s, token_set_res, tax.canon) or (s in token_set_res)) for s in skills) / len(skills)

    hard_cov = _cov(required_hard)
    soft_cov = _cov(required_soft)
    nth_cov = _cov(nice_to_have)

    # Families: if JD says "cloud" family, allow aws/gcp/azure to satisfy (if present in families)
    for fam, members in (tax.families or {}).items():
        if fam in required_hard and any(m in token_set_res for m in members):
            hard_cov = min(1.0, hard_cov + 1.0 / max(1, len(required_hard)))

    # Phrases overlap
    jd_bigrams = bigrams(tokenize(jd_text))
    phrase_cov = (sum(1 for p in jd_bigrams if p in resume_text.lower()) / max(1, len(jd_bigrams))) if jd_bigrams else 0.0

    # Sections presence/order
    present_sections = {k: (k in sections and len(sections[k]) > 40) for k in [
        "summary","experience","projects","skills","education","certifications","achievements","links",
    ]}
    order_penalty = 0.0
    logical_order = ["summary", "skills", "experience", "projects", "education"]
    indices = [list(sections.keys()).index(s) for s in logical_order if s in sections]
    if indices != sorted(indices):
        order_penalty = 0.05

    # Impact bullets
    bullets = re.findall(r"(?:^|\n)[•\-–]\s*(.+?)(?=\n|$)", resume_text)
    digits_ratio = (sum(bool(re.search(r"\d", b)) for b in bullets) / len(bullets)) if bullets else 0.0

    # Quality
    q = readability_metrics(resume_text)
    grammar_count = grammar_issues(resume_text) if args.grammar_check else 0
    hygiene_score = max(0.0, 1.0 - min(1.0, grammar_count / 50.0))
    read_norm = 0.0
    if q["flesch_reading_ease"]:
        read_norm = min(1.0, max(0.0, (q["flesch_reading_ease"] / 100.0)))

    # Seniority & role scope
    res_seniority = detect_seniority(resume_text, tax.seniority_terms)
    sdist = seniority_distance(res_seniority, jd_seniority)
    seniority_fit = max(0.0, 1.0 - min(1.0, sdist / 4.0))
    res_scope = detect_role_scope(resume_text, tax.role_scope_terms)
    scope_fit = 1.0 if (jd_scope == res_scope) else 0.6 if (jd_scope in ("lead","manager") and res_scope == "ic") else 0.8

    # Methodologies / Domain coverage
    meth_cov = _cov(jd_methodologies)
    domain_cov = _cov(jd_domains)

    # Breadth vs depth
    breadth, depth = compute_breadth_depth(required_hard, per_skill_years_est)

    # Placement score (summary/experience vs skills list)
    placement = keyword_placement_score(sections, required_hard, tax.canon)

    # Recency score: exponential decay with half-life
    half_life = max(1, int(thresholds.get("recency_half_life_months", 24)))
    def _rec_norm(m: Optional[int]) -> float:
        if m is None: return 0.0
        return 0.5 ** (m / half_life)  # 1.0 now, 0.5 at half_life months, etc.
    rec_vals = [_rec_norm(per_skill_last_used.get(s)) for s in required_hard] or [1.0]
    recency = sum(rec_vals) / len(rec_vals)

    # Education & certs
    edu_score, edu_meta = edu_certs_score(sections, tax)

    # Stability & chronology
    stability_score, stab_meta = stability_progression_score(exps)
    chrono = chronology_consistency(exps)

    # Contact & links
    contacts = contact_links(resume_text)
    contact_ok = bool(contacts["emails"]) and (bool(contacts["phones"]) or contacts["linkedin"] or contacts["github"])

    # Formatting & anti-gaming
    fmt_flags = {
        "multi_column_pdf": bool(pdf_path and detect_multicolumn(pdf_path)),
    }
    ag = anti_gaming_metrics(resume_text, pdf_path)
    anti_gaming_penalty = 0.0
    if ag["top_token_ratio"] > 0.06 or ag["bigram_unique_ratio"] < 0.4 or ag["white_text_ratio"] > 0.01 or ag["tiny_font_ratio"] > 0.01:
        anti_gaming_penalty = 0.05
    penalties = 0.02 * sum(v for v in fmt_flags.values()) + anti_gaming_penalty + order_penalty
    penalties = min(0.20, penalties)  # cap penalty to avoid overkill

    # Compliance gates
    location_req = extract_location_req(jd_text)

    loc_ok = True if not location_req else (location_req in resume_text.lower() or "remote" in jd_text.lower())
    auth_ok = True if not work_auth_req else (work_auth_req in resume_text.lower())
    degree_ok = True if not degree_req else (degree_req in resume_text.lower())
    seniority_gate = sdist <= 3

    lacking_years = {k: v for k, v in per_skill_years_req.items() if per_skill_years_est.get(k, 0.0) < v}
    req_coverage = (sum(synonym_hit(r, token_set_res, tax.canon) or (r in token_set_res) for r in required_hard) / max(1, len(required_hard))) if required_hard else 1.0

    gates = {
        "required_skills_coverage_ok": req_coverage >= (args.min_req_coverage if required_hard else 0.0),
        "per_skill_years_ok": len(lacking_years) == 0,
        "total_years_ok": years_total >= min_years_required,
        "location_ok": loc_ok,
        "work_auth_ok": auth_ok,
        "degree_ok": degree_ok,
        "seniority_ok": seniority_gate,
        "contact_ok": contact_ok,
        "chronology_ok": chrono["gaps_over_6m"] == 0,
    }

    # Scores per component (all 0..1)
    s_hard = 0.55 * req_coverage + 0.20 * nth_cov + 0.25 * phrase_cov
    s_soft = soft_cov
    s_semantic = 0.5 * sem.get("bi_encoder_score", 0.0) + 0.5 * sem.get("cross_encoder_score", 0.0)
    s_experience_total = min(1.0, years_total / max(1.0, (min_years_required or 1.0)))
    # per-skill coverage
    per_skill_cov = 1.0
    if per_skill_years_req:
        coverages = []
        for k, need in per_skill_years_req.items():
            have = per_skill_years_est.get(k, 0.0)
            coverages.append(min(1.0, have / max(need, 0.0001)))
        per_skill_cov = sum(coverages) / len(coverages)
    s_per_skill_years = per_skill_cov
    s_recency = recency
    s_methodologies = meth_cov
    s_domain = domain_cov
    s_breadth_depth = 0.5 * breadth + 0.5 * depth
    s_impact = digits_ratio
    s_placement = placement
    s_quality = 0.6 * read_norm + 0.4 * hygiene_score
    s_seniority = seniority_fit
    s_role_scope = scope_fit
    s_stability = stability_score
    s_compliance = (0.4 * (1.0 if loc_ok else 0.0) +
                    0.2 * (1.0 if auth_ok else 0.0) +
                    0.2 * (1.0 if degree_ok else 0.0) +
                    0.2 * (1.0 if seniority_gate else 0.0))

    # Weighted sum
    comps = {
        "hard_keywords": s_hard, "soft_keywords": s_soft, "semantic": s_semantic,
        "experience_total": s_experience_total, "per_skill_years": s_per_skill_years,
        "recency": s_recency, "methodologies": s_methodologies, "domain": s_domain,
        "breadth_depth": s_breadth_depth, "impact": s_impact, "placement": s_placement,
        "quality": s_quality, "seniority": s_seniority, "role_scope": s_role_scope,
        "stability": s_stability, "compliance": s_compliance
    }
    # Normalize weights to sum to 1.0 (ignore penalties here)
    wsum = sum(max(0.0, weights.get(k, 0.0)) for k in comps.keys()) or 1.0
    score_raw = sum((weights.get(k, 0.0) / wsum) * max(0.0, min(1.0, v)) for k, v in comps.items())
    score = round(max(0, min(100, (score_raw - penalties) * 100)))

    # Component scores (0..100)
    component_scores = {k: round(max(0.0, min(1.0, v)) * 100) for k, v in comps.items()}
    component_scores["formatting_penalty_pct"] = round(penalties * 100, 1)

    # Suggestions
    suggestions: List[str] = []
    missing_required = [r for r in required_hard if not (synonym_hit(r, token_set_res, tax.canon) or (r in token_set_res))]
    if missing_required:
        suggestions.append("Add concrete evidence for missing hard skills: " + ", ".join(sorted(missing_required)) + ".")
    if lacking_years:
        gaps = sorted(((k, per_skill_years_req[k] - per_skill_years_est.get(k, 0.0)) for k in lacking_years), key=lambda x: -x[1])
        suggestions.append("Strengthen timeline evidence: " + ", ".join(f"{k} (+{gap:.1f}y)" for k, gap in gaps))
    if s_recency < 0.5:
        suggestions.append("Surface recent projects using the required stack (last 12–24 months) near the top.")
    if s_impact < 0.5:
        suggestions.append("Increase quantified outcomes in bullets (%, time, cost, revenue, latency, users).")
    if s_placement < 0.6:
        suggestions.append("Mention core JD skills in Summary and Experience, not only in the Skills list.")
    if not contacts["emails"] or (not contacts["phones"] and not contacts["linkedin"] and not contacts["github"]):
        suggestions.append("Add a reachable email and at least one of: phone, LinkedIn, or GitHub.")
    if chrono["gaps_over_6m"] > 0:
        suggestions.append("Address employment gaps ≥ 6 months (brief explanation or projects/education).")
    if not gates["seniority_ok"]:
        suggestions.append("Align title/scope to JD seniority; emphasize leadership/ownership if relevant.")
    if meth_cov < 0.5 and jd_methodologies:
        suggestions.append("Call out methodologies (e.g., microservices, CI/CD, testing) in bullets with examples.")
    if domain_cov < 0.5 and jd_domains:
        suggestions.append("Highlight domain experience (e.g., fintech/adtech) with products, scale, or regulations.")

    t1 = time.perf_counter()
    report: Dict[str, Any] = {
        "version": VERSION,
        "role_name": role_name,
        "overall_score": score,
        "component_scores": component_scores,
        "gates": gates,
        "years_total_detected": years_total,
        "required_coverage": round(req_coverage, 3),
        "nice_to_have_coverage": round(nth_cov, 3),
        "phrase_coverage": round(phrase_cov, 3),
        "min_req_coverage": args.min_req_coverage,
        "missing_required_skills": missing_required,
        "per_skill_years_required": per_skill_years_req,
        "per_skill_years_estimated": per_skill_years_est,
        "per_skill_last_used_months": per_skill_last_used,
        "sections_present": present_sections,
        "format_flags": fmt_flags,
        "quality": {"readability": q, "grammar_issues": grammar_count},
        "anti_gaming": ag,
        "semantic": sem,
        "contacts": contacts,
        "stability": stab_meta,
        "chronology": chrono,
        "education": edu_meta,
        "suggestions": suggestions,
        "timings_s": {"scoring": round(t1 - t0, 3)},
    }
    return report

# ---------- Evaluation ----------
def evaluate_single(args: argparse.Namespace, tax: TaxonomyBundle, weights: Dict[str, float], thresholds: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
    role: Dict[str, Any] = {}
    if args.role and pathlib.Path(args.role).exists():
        try:
            role = json.loads(read_textfile(args.role))
        except Exception:
            role = {}

    jd_text = read_textfile(args.jd) or role.get("job_description", "")
    active_tax = subset_taxonomy_for_jd(jd_text, tax) if args.jd_skill_filter else tax

    if not role.get("required_skills") and jd_text:
        auto = extract_requirements_from_jd(jd_text, active_tax)
        role.setdefault("role_name", auto.role_name)
        role.setdefault("required_skills", auto.required_skills)
        role.setdefault("nice_to_have_skills", auto.nice_to_have_skills)
        role.setdefault("per_skill_years", auto.per_skill_years)
        role.setdefault("min_years_experience_total", auto.min_years_total)
        role.setdefault("degree", auto.degree)
        role.setdefault("location", auto.location)
        role.setdefault("work_authorization", auto.work_authorization)
        role.setdefault("seniority", auto.seniority)
        role.setdefault("soft_required", auto.soft_required)
        role.setdefault("domains", auto.domains)
        role.setdefault("methodologies", auto.methodologies)
        role.setdefault("role_scope", auto.role_scope)

    resume_text = extract_pdf_text(args.pdf, use_ocr=args.use_ocr) if args.pdf else ""
    resume_text = normalize_dates_text(resume_text)

    if not resume_text.strip():
        resume_text = strip_latex(read_textfile(args.tex)) if args.tex else ""
    resume_text = normalize_bullets(normalize_ws(resume_text))

    sections = split_sections(resume_text)
    exps = segment_experiences(resume_text)

    report = compute_scores(jd_text, resume_text, sections, exps, active_tax, role, args, args.pdf, weights, thresholds)
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
        if report.get("per_skill_years_required"):
            lacking = {k: v for k, v in report["per_skill_years_required"].items() if report["per_skill_years_estimated"].get(k,0.0) < v}
            if lacking:
                lines.append("- ⚠️ Insufficient per-skill years: " + ", ".join(f"{k} (need {v}y)" for k, v in lacking.items()))
        lines.append("\n**Breakdown**")
        lines.append(
            f"- Hard: {cs['hard_keywords']} | Soft: {cs['soft_keywords']} | Semantic: {cs['semantic']} | "
            f"Experience: {cs['experience_total']} | Per-skill: {cs['per_skill_years']} | Recency: {cs['recency']} | "
            f"Methodologies: {cs['methodologies']} | Domain: {cs['domain']} | Breadth/Depth: {cs['breadth_depth']} | "
            f"Impact: {cs['impact']} | Placement: {cs['placement']} | Quality: {cs['quality']} | "
            f"Seniority: {cs['seniority']} | Role Scope: {cs['role_scope']} | Stability: {cs['stability']} | "
            f"Compliance: {cs['compliance']} | Formatting penalty: -{cs['formatting_penalty_pct']}%\n"
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

def evaluate_batch(args: argparse.Namespace, tax: TaxonomyBundle, weights: Dict[str, float], thresholds: Dict[str, Any]) -> None:
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
    active_tax = subset_taxonomy_for_jd(jd_text, tax) if args.jd_skill_filter else tax

    if not role.get("required_skills") and jd_text:
        auto = extract_requirements_from_jd(jd_text, active_tax)
        role = {**auto.__dict__, **role}

    rows: List[Dict[str, Any]] = []
    jsonl_path = pathlib.Path(args.out_jsonl) if args.out_jsonl else None
    jsonl_fh = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None

    for pdf in files:
        logger.info(f"Scoring: {pdf.name}")
        args_single = argparse.Namespace(**vars(args))
        args_single.pdf = str(pdf)
        args_single.tex = None
        report, resume_text, jd_text2 = evaluate_single(args_single, active_tax, weights, thresholds)
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

# ---------- CLI ----------
def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=f"ATS Pro Plus v{VERSION}")
    ap.add_argument("--pdf", help="Path to compiled resume PDF")
    ap.add_argument("--tex", help="Path to LaTeX source (fallback)")
    ap.add_argument("--jd", default="job-description.txt", help="Path to JD text file (optional)")
    ap.add_argument("--role", default="", help="Path to role JSON (optional)")
    ap.add_argument("--taxonomy", default="", help="Path to taxonomy JSON (required for rich scoring)")
    ap.add_argument("--scoring-config", default="", help="Path to scoring config JSON (optional)")
    ap.add_argument("--jd-skill-filter", action="store_true",
                    help="Use only taxonomy skills that actually appear in the JD for extraction and scoring")
    ap.add_argument("--out-json", default="ats-pro-report.json")
    ap.add_argument("--out-md", default="ats-pro-report.md")
    ap.add_argument("--out-html", default="ats-pro-report.html")
    ap.add_argument("--min-score", type=int, default=75)
    ap.add_argument("--min-req-coverage", type=float, default=0.85)
    ap.add_argument("--use-ocr", action="store_true", help="Enable OCR fallback for scanned PDFs")
    ap.add_argument("--grammar-check", action="store_true", help="Run grammar check (language_tool_python)")
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
    tax = load_taxonomy(args.taxonomy)
    weights, thresholds = load_scoring_config(args.scoring_config)

    if args.res_dir:
        evaluate_batch(args, tax, weights, thresholds)
        return

    if not args.pdf and not args.tex:
        logger.error("Provide at least --pdf or --tex for single evaluation, or --res-dir for batch.")
        sys.exit(2)

    report, resume_text, jd_text = evaluate_single(args, tax, weights, thresholds)
    save_outputs_single(report, args, resume_text, jd_text)

if __name__ == "__main__":
    main()
