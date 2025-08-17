from __future__ import annotations
import re, math
from typing import Dict, List, Tuple, Optional

ACTION_VERBS = {
    "led","owned","built","designed","developed","implemented","launched","scaled","optimized","reduced","increased",
    "migrated","automated","refactored","delivered","shipped","architected","mentored","drove","improved","created"
}

SECTION_HINTS = ["experience", "education", "projects", "skills", "summary", "publications", "awards"]

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def _semantic_similarity(resume_text: str, jd_text: str, nlp, res_sents: List[str], k: int = 5) -> Tuple[float, List[str]]:
    """
    If spaCy with word vectors is available: doc.similarity.
    Else: fallback to Jaccard on tokens and pick top-K numeric/keyword-rich sentences as 'snippets'.
    """
    if nlp:
        try:
            rdoc = nlp(resume_text)
            jdoc = nlp(jd_text)
            sim = float(rdoc.similarity(jdoc))
            # pick informative snippets (with numbers or verbs)
            sents = sorted(res_sents, key=lambda s: (sum(ch.isdigit() for ch in s), -len(s)), reverse=True)[:k]
            return sim, sents
        except Exception:
            pass
    # fallback
    sim = _jaccard(resume_text.lower().split(), jd_text.lower().split())
    sents = sorted(res_sents, key=lambda s: (sum(ch.isdigit() for ch in s), -len(s)), reverse=True)[:k]
    return float(sim), sents

def _digits_ratio(text: str) -> float:
    tot = max(1, sum(ch.isdigit() for ch in text) + 0)
    return tot / max(1, len(text))

def _readability_norm(text: str) -> float:
    # simple: prefer 10-24 words per sentence
    sents = re.split(r"[.!?]\s+", text)
    if not sents:
        return 1.0
    scores = []
    for s in sents:
        words = len(s.split())
        if words == 0:
            continue
        # ideal ~16 words -> score 1.0, else drop with quadratic loss
        delta = (words - 16) / 16.0
        scores.append(max(0.0, 1.0 - (delta * delta)))
    return sum(scores) / max(1, len(scores))

def _sections_score(text: str) -> float:
    low = text.lower()
    hits = sum(1 for h in SECTION_HINTS if h in low)
    return min(1.0, hits / 5.0)

def _seniority_ok(resume_text: str, hint: Optional[str]) -> bool:
    if not hint:
        return True
    low = resume_text.lower()
    if hint in ["principal", "staff"]:
        return any(w in low for w in ["principal","staff","lead","architect"])
    if hint == "lead":
        return any(w in low for w in ["lead","principal","staff"])
    if hint == "senior":
        return any(w in low for w in ["senior","lead","principal","staff"])
    if hint == "junior":
        return not any(w in low for w in ["senior","lead","principal","staff"])
    return True

def _impact_score(text: str) -> float:
    digits = sum(ch.isdigit() for ch in text)
    tokens = max(1, len(text.split()))
    ratio = digits / tokens
    verbs = sum(1 for t in text.lower().split() if t.strip(",.") in ACTION_VERBS)
    verb_ratio = verbs / tokens
    # scale to 0..1
    return min(1.0, (ratio * 6.0) + (verb_ratio * 10.0)), ratio

def _experience_score(per_skill_years_est: Dict[str,float], per_skill_years_req: Dict[str,float]) -> Tuple[float, Dict[str,float]]:
    lacking = {}
    scores = []
    for skill, req in per_skill_years_req.items():
        est = per_skill_years_est.get(skill, 0.0)
        cov = min(1.0, est / max(0.1, req))
        scores.append(cov)
        if est < req:
            lacking[skill] = round(req,2)
    return (sum(scores)/max(1,len(scores))) if scores else 1.0, lacking

def compute_component_scores(
    resume_text: str,
    jd_text: str,
    res_tokens: List[str],
    jd_tokens: List[str],
    res_sents: List[str],
    nlp,
    per_skill_years_est: Dict[str,float],
    per_skill_years_req: Dict[str,float],
    kw_summary: Dict,
    kw_details: List[Dict],
    snippets: int = 5,
    fast: bool = False,
    grammar_check: bool = False,
    seniority_hint: Optional[str] = None,
):
    # Keywords score
    kw_score = (0.7 * kw_summary["required"]["coverage"] + 0.3 * kw_summary["nice"]["coverage"]) * 100.0

    # Semantic similarity
    sem_raw, top_snips = _semantic_similarity(resume_text, jd_text, nlp, res_sents, k=snippets)
    sem_score = min(100.0, max(0.0, sem_raw * 100.0))

    # Sections score
    sec_score = _sections_score(resume_text) * 100.0

    # Experience coverage
    exp_cov, lacking_years = _experience_score(per_skill_years_est, per_skill_years_req)
    exp_score = exp_cov * 100.0

    # Impact
    impact_norm, digits_ratio = _impact_score(resume_text)
    imp_score = impact_norm * 100.0

    # Quality (readability + optional grammar)
    readability_norm = _readability_norm(resume_text)
    quality = readability_norm
    if grammar_check and not fast:
        try:
            import language_tool_python  # optional
            tool = language_tool_python.LanguageTool("en-US")
            errs = tool.check(resume_text[:40000])  # cap for speed
            # penalize if many issues; 0.9 base * exp(-errs/300)
            quality *= math.exp(-len(errs)/300.0)
        except Exception:
            pass
    quality_score = min(100.0, max(0.0, quality * 100.0))

    # Seniority
    seniority_ok = _seniority_ok(resume_text, seniority_hint)
    seniority_score = 100.0 if seniority_ok else 70.0

    # Formatting/anti-gaming penalty placeholder (0 if text-only extraction)
    anti_gaming = {"white_text_ratio": 0.0, "tiny_font_ratio": 0.0}
    formatting_penalty_pct = 0.0

    # Overall score weights
    weights = {
        "keywords": 0.24, "semantic": 0.18, "sections": 0.10, "experience": 0.18,
        "impact": 0.12, "quality": 0.10, "seniority": 0.08
    }
    overall = (
        kw_score * weights["keywords"] +
        sem_score * weights["semantic"] +
        sec_score * weights["sections"] +
        exp_score * weights["experience"] +
        imp_score * weights["impact"] +
        quality_score * weights["quality"] +
        seniority_score * weights["seniority"]
    )
    overall = max(0.0, overall - formatting_penalty_pct)

    component_scores = {
        "keywords": round(kw_score, 1),
        "semantic": round(sem_score, 1),
        "sections": round(sec_score, 1),
        "experience": round(exp_score, 1),
        "impact": round(imp_score, 1),
        "quality": round(quality_score, 1),
        "seniority": round(seniority_score, 1),
        "formatting_penalty_pct": round(formatting_penalty_pct, 1),
    }

    diagnostics = {
        "overall_score": round(overall, 1),
        "digits_ratio": round(digits_ratio, 4),
        "readability_norm": round(readability_norm, 3),
        "lacking_years": lacking_years,
        "top_snippets": top_snips,
        "anti_gaming": anti_gaming,
        "seniority_ok": seniority_ok,
    }
    return component_scores, diagnostics

def compute_gates(
    overall_score: float,
    required_coverage: float,
    per_skill_years_est: Dict[str,float],
    per_skill_years_req: Dict[str,float],
    seniority_ok: bool,
    min_score: float,
    min_req_cov: float,
    anti_gaming_flags: Dict[str, float],
):
    years_ok = True
    for skill, req in per_skill_years_req.items():
        if per_skill_years_est.get(skill, 0.0) + 1e-6 < req:
            years_ok = False
            break

    ag_ok = all((anti_gaming_flags or {}).get(k, 0.0) < 0.02 for k in ["white_text_ratio", "tiny_font_ratio"])

    gates = {
        "min_score_ok": bool(overall_score >= min_score),
        "req_keywords_ok": bool(required_coverage >= min_req_cov),
        "years_ok": years_ok,
        "seniority_ok": seniority_ok,
        "anti_gaming_ok": ag_ok,
    }
    return gates

def build_suggestions(
    kw_missing_ranked: List[Dict],
    nice_missing_ranked: List[Dict],
    lacking_years: Dict[str,float],
    per_skill_years_est: Dict[str,float],
    digits_ratio: float,
    readability_norm: float,
    gates: Dict[str,bool],
    seniority_ok: bool,
) -> List[str]:
    sug: List[str] = []

    # Required keyword gaps
    req_miss = [d["term"] for d in kw_missing_ranked if d.get("type") == "required"][:6]
    if req_miss:
        sug.append("Add evidence for REQUIRED skills: " + ", ".join(req_miss) + ".")

    nice_miss = [d["term"] for d in nice_missing_ranked][:5]
    if nice_miss:
        sug.append("Weave in NICE-TO-HAVE skills (if applicable): " + ", ".join(nice_miss) + ".")

    if lacking_years:
        gaps = []
        for k, req in lacking_years.items():
            est = per_skill_years_est.get(k, 0.0)
            if est < req:
                gaps.append((k, req - est))
        gaps.sort(key=lambda x: -x[1])
        if gaps:
            sug.append("Quantify experience to close gaps: " + ", ".join(f"{k} (+{gap:.1f}y)" for k, gap in gaps[:5]))

    if digits_ratio < 0.0008:
        sug.append("Increase metrics in bullets (%, $, time); target >50% bullets with numbers.")
    if readability_norm < 0.4:
        sug.append("Improve readability (short sentences, active voice, consistent tense).")
    if not seniority_ok:
        sug.append("Align scope/title with JD seniority; foreground leadership/impact.")

    failed = [k for k, v in gates.items() if not v]
    if failed:
        pretty = [f.replace("_"," ") for f in failed]
        sug.append("Fix failing checks: " + ", ".join(pretty) + ".")
    if not sug:
        sug.append("✅ Looks good!")
    return sug
