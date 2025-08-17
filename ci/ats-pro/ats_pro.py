#!/usr/bin/env python3
import os, re, json, math, pathlib, collections, argparse
from datetime import datetime
import dateparser
from rapidfuzz import fuzz
from textstat import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text
import pdfplumber

# ---------- helpers ----------
def read_textfile(p):
    p = pathlib.Path(p)
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

def extract_pdf_text(pdf_path):
    try:
        text = extract_text(str(pdf_path)) or ""
    except Exception:
        text = ""
    return text

def detect_multicolumn(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                xs = [w["x0"] for w in words] if words else []
                if xs:
                    # lots of distinct x starts → likely tables/columns
                    buckets = set(int(x//10) for x in xs)
                    if len(buckets) > 30:
                        return True
    except Exception:
        pass
    return False

def strip_latex(tex_src: str) -> str:
    s = re.sub(r"%.*", "", tex_src)
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def split_sections(text):
    lower = text.lower()
    heads = {
      "summary": r"(?:summary|objective)",
      "experience": r"(?:experience|employment|work experience)",
      "education": r"(?:education|academics)",
      "projects": r"(?:projects?)",
      "skills": r"(?:skills?|technologies|tech stack)",
      "certifications": r"(?:certifications?|licenses?)",
      "achievements": r"(?:awards?|achievements?)",
      "publications": r"(?:publications?)",
      "links": r"(?:links|profiles)"
    }
    spans, indices = {}, []
    for name, pat in heads.items():
        m = re.search(rf"\b{pat}\b", lower)
        if m: indices.append((m.start(), name))
    indices.sort()
    indices.append((len(lower), "_end"))
    for i in range(len(indices)-1):
        start, name = indices[i]
        end, _ = indices[i+1]
        spans[name] = lower[start:end]
    return spans

def tokenize(s):
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+.#-]{2,}", s.lower())

def bigrams(tokens):
    return {" ".join(p) for p in zip(tokens, tokens[1:])}

def synonym_hit(term, tokenset, synmap):
    t = term.lower()
    if t in tokenset: return True
    for syn in synmap.get(t, set()):
        if any(fuzz.partial_ratio(syn, r) >= 90 for r in tokenset):
            return True
    return False

def parse_years(text):
    rng_pat = re.compile(r"([A-Za-z]{3,9}[' ]?\d{2,4}|\d{4})\s*[–-]\s*(Present|\d{4}|[A-Za-z]{3,9}[' ]?\d{2,4})", re.I)
    now = datetime.now()
    months_total = 0
    for m in rng_pat.finditer(text):
        a, b = m.group(1), m.group(2)
        da = dateparser.parse(a, settings={"PREFER_DAY_OF_MONTH": "first"})
        db = now if re.match(r"present", b, re.I) else dateparser.parse(b, settings={"PREFER_DAY_OF_MONTH": "first"})
        if da and db and db > da:
            months_total += (db.year - da.year)*12 + (db.month - da.month)
    return round(months_total/12.0, 2)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to compiled resume PDF")
    ap.add_argument("--tex", required=True, help="Path to LaTeX source (fallback parsing & format flags)")
    ap.add_argument("--jd", default="job-description.txt", help="Path to JD text file (optional)")
    ap.add_argument("--role", default="", help="Path to role JSON (optional)")
    ap.add_argument("--out-json", default="ats-pro-report.json")
    ap.add_argument("--out-md", default="ats-pro-report.md")
    ap.add_argument("--min-score", type=int, default=75)
    ap.add_argument("--min-req-coverage", type=float, default=0.85)
    args = ap.parse_args()

    # Load role/JD
    role = {}
    if args.role and pathlib.Path(args.role).exists():
        try:
            role = json.loads(read_textfile(args.role))
        except Exception:
            role = {}
    jd_text = read_textfile(args.jd) or role.get("job_description","")
    required = set([x.lower() for x in role.get("required_skills", [])])
    nice_to_have = set([x.lower() for x in role.get("nice_to_have_skills", [])])
    min_years_required = float(role.get("min_years_experience_total", 0))
    per_skill_years = {k.lower(): float(v) for k,v in role.get("per_skill_years", {}).items()}
    location_req = role.get("location", "").lower()
    work_auth_req = role.get("work_authorization", "").lower()
    degree_req = role.get("degree", "").lower()
    role_name = role.get("role_name","unspecified")

    # Resume text (prefer PDF)
    resume_text = extract_pdf_text(args.pdf)
    if not resume_text.strip():
        resume_text = strip_latex(read_textfile(args.tex))
    resume_text = re.sub(r"\s+", " ", resume_text).strip()

    # Sections & order
    sections = split_sections(resume_text)
    present_sections = {k: (k in sections and len(sections[k])>40) for k in
                        ["summary","experience","projects","skills","education","certifications","achievements","links"]}
    order_penalty = 0.0
    logical_order = ["summary","skills","experience","projects","education"]
    indices = [list(sections.keys()).index(s) for s in logical_order if s in sections]
    if indices != sorted(indices):
        order_penalty = 0.05

    # Years of experience
    years_total = parse_years(resume_text)

    # Tokens & phrases
    jd_tokens = tokenize(jd_text)
    res_tokens = tokenize(resume_text)
    token_set_res = set(res_tokens)
    jd_bigrams = bigrams(jd_tokens)

    # Embeddings + TF-IDF similarities
    sem_sim = 0.0
    if jd_text.strip():
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        sentences = re.split(r"(?<=[.!?])\s+", resume_text)
        emb_resume = model.encode(sentences or [resume_text], normalize_embeddings=True, show_progress_bar=False)
        emb_jd = model.encode([jd_text], normalize_embeddings=True, show_progress_bar=False)
        sims = cosine_similarity(emb_jd, emb_resume)[0]
        sem_sim = float(max(sims))
    tfidf_sim = 0.0
    if jd_text.strip():
        tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        tfidf_mat = tfidf.fit_transform([jd_text, resume_text])
        tfidf_sim = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])

    # Synonyms map (extend as needed)
    SYN = {
      "c++": {"cpp","c plus plus"},
      "postgresql": {"postgres","psql"},
      "aws": {"amazon web services"},
      "gcp": {"google cloud"},
      "ml": {"machine learning"},
      "react.js": {"react","reactjs"},
      "node.js": {"node","nodejs"},
      "kubernetes": {"k8s"},
      "ci/cd": {"ci cd","continuous integration","continuous delivery"},
      "object-oriented": {"oop","object oriented"}
    }

    # Coverage & gates
    req_hits = {r: (synonym_hit(r, token_set_res, SYN) or (r in token_set_res)) for r in required}
    req_coverage = (sum(req_hits.values())/max(1,len(required))) if required else 1.0
    nth_hits = {n: (synonym_hit(n, token_set_res, SYN) or (n in token_set_res)) for n in nice_to_have}
    nth_coverage = (sum(nth_hits.values())/max(1,len(nice_to_have))) if nice_to_have else 0.0
    phrase_cov = (sum(1 for p in jd_bigrams if p in resume_text.lower())/max(1,len(jd_bigrams))) if jd_bigrams else 0.0

    per_skill_years_est = {}
    for skill, yrs_req in per_skill_years.items():
        present = synonym_hit(skill, token_set_res, SYN) or (skill in token_set_res)
        per_skill_years_est[skill] = years_total * (0.6 if present else 0.0)  # conservative proxy

    loc_ok   = True if not location_req else (location_req in resume_text.lower())
    auth_ok  = True if not work_auth_req else (work_auth_req in resume_text.lower())
    degree_ok= True if not degree_req else (degree_req in resume_text.lower())

    # Hygiene (bullets, numbers, stuffing, typos-light)
    bullets = re.findall(r"(?:^|\n)[•\-–]\s*(.+?)(?=\n|$)", resume_text)
    digits_ratio = (sum(bool(re.search(r"\d", b)) for b in bullets)/len(bullets)) if bullets else 0.0
    counts = collections.Counter(res_tokens)
    total_tokens = max(1,len(res_tokens))
    top5 = counts.most_common(5)
    stuffing = any(c/total_tokens > 0.04 and t not in {"and","the","with","for"} for t,c in top5)
    try:
        from spellchecker import SpellChecker
        sc = SpellChecker(distance=1)
        sample = [w for w in res_tokens if w.isalpha() and len(w)>2 and not re.match(r"[A-Za-z]\+\+|c\+\+", w)]
        miss = sc.unknown(sample[:400])
        typo_rate = len(miss)/max(1,len(sample[:400]))
    except Exception:
        typo_rate = 0.0

    # Formatting flags from LaTeX + PDF
    tex_src = read_textfile(args.tex)
    fmt_flags = {
      "multi_column_pdf": detect_multicolumn(args.pdf),
      "tables_or_tabular": bool(re.search(r"\\begin\{table\}|\\tabular|\\begin\{longtable\}", tex_src)),
      "multicol_env": bool(re.search(r"\\begin\{multicols?\}", tex_src)),
      "images": bool(re.search(r"\\includegraphics", tex_src)),
      "headers_footers": bool(re.search(r"\\pagestyle|\\fancyhdr|\\header|\\footer", tex_src)),
      "icons_colors": bool(re.search(r"\\fa[ A-Za-z]|\\textcolor|\\color\{|xcolor", tex_src)),
      "footnotes": bool(re.search(r"\\footnote", tex_src)),
    }
    penalties = sum(fmt_flags.values()) * 0.02  # 2% each

    # Gates
    gates = {
      "required_skills_coverage_ok": req_coverage >= (args.min_req_coverage if required else 0.0),
      "total_years_ok": years_total >= min_years_required,
      "location_ok": loc_ok,
      "work_auth_ok": auth_ok,
      "degree_ok": degree_ok
    }

    # Scoring
    s_keywords  = 0.40 * (0.5*req_coverage + 0.3*nth_coverage + 0.2*phrase_cov)
    s_semantic  = 0.20 * ((sem_sim + tfidf_sim)/2.0)
    s_sections  = 0.12 * ((sum(present_sections.values())/len(present_sections)) - 0.0 if not order_penalty else (sum(present_sections.values())/len(present_sections) - order_penalty))
    s_experience= 0.12 * min(1.0, years_total / max(1.0, min_years_required or 1.0))
    s_impact    = 0.08 * digits_ratio
    s_hygiene   = 0.08 * (1.0 - min(1.0, typo_rate*4.0))
    score_raw = (s_keywords + s_semantic + s_sections + s_experience + s_impact + s_hygiene)
    score = round(max(0, min(100, (score_raw - penalties) * 100)))

    missing_required = [k for k,v in req_hits.items() if not v]
    lacking_years = {k:v for k,v in per_skill_years.items() if per_skill_years_est.get(k,0.0) < v}

    report = {
      "role_name": role_name,
      "overall_score": score,
      "component_scores": {
        "keywords": round(s_keywords*100),
        "semantic": round(s_semantic*100),
        "sections": round(s_sections*100),
        "experience": round(s_experience*100),
        "impact": round(s_impact*100),
        "hygiene": round(s_hygiene*100),
        "formatting_penalty_pct": round(penalties*100,1)
      },
      "gates": gates,
      "years_total_detected": years_total,
      "required_coverage": round(req_coverage,3),
      "nice_to_have_coverage": round(nth_coverage,3),
      "phrase_coverage": round(phrase_cov,3),
      "missing_required_skills": missing_required,
      "per_skill_years_required": per_skill_years,
      "per_skill_years_estimated": per_skill_years_est,
      "lacking_years": lacking_years,
      "sections_present": present_sections,
      "format_flags": fmt_flags,
      "stuffing_detected": stuffing,
      "typo_rate_sampled": round(typo_rate,3),
      "notes": [
        "Analyzes compiled PDF (ATS realistic). Gates must pass; score must meet min-score.",
        "Keyword stuffing, heavy formatting, or multi-column layouts reduce score."
      ]
    }

    pathlib.Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    def b(v): return "✅" if v else "❌"
    lines = []
    lines.append("### 🧠 ATS Pro Report (Strict)")
    lines.append(f"**Overall:** **{score}/100**")
    lines.append("")
    lines.append("**Gates**")
    for k,v in gates.items():
        lines.append(f"- {b(v)} {k.replace('_',' ').title()}")
    if missing_required:
        lines.append(f"- ⚠️ Missing required skills: {', '.join(sorted(missing_required))}")
    if lacking_years:
        lines.append(f"- ⚠️ Insufficient years for: " + ", ".join(f"{k} (need {v}y)" for k,v in lacking_years.items()))
    lines.append("")
    cs = report["component_scores"]
    lines.append("**Breakdown**")
    lines.append(f"- Keywords: {cs['keywords']} | Semantic: {cs['semantic']} | Sections: {cs['sections']} | Experience: {cs['experience']} | Impact: {cs['impact']} | Hygiene: {cs['hygiene']} | Formatting penalty: -{cs['formatting_penalty_pct']}%")
    lines.append("")
    lines.append("**Format flags**")
    for k,v in fmt_flags.items():
        lines.append(f"- {'⚠️' if v else '✅'} {k.replace('_',' ')}")
    lines.append("")
    lines.append(f"_Role profile_: **{role_name}**" if role_name else "_Role profile_: (not provided)")
    pathlib.Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")

    # CLI exit for CI
    all_gates = all(gates.values())
    if (not all_gates) or (score < args.min_score):
        # Non-zero exit to fail the job in CI
        print(json.dumps(report, indent=2))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
