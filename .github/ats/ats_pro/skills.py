from __future__ import annotations
import re, collections
from typing import Dict, Set, Tuple, List, Optional

TECHLIKE = re.compile(r"""
    (?ix)
    (?:c\+\+|c#|\.net|node\.js|react\.js|react|angular|vue|spring|django|flask|kotlin|swift|
     typescript|javascript|java|python|go|golang|rust|sql|nosql|redis|kafka|spark|hadoop|aws|gcp|azure|
     docker|kubernetes|k8s|terraform|ansible|git|linux|rest|grpc|microservices|mongodb|postgres|mysql)
""")

def build_taxonomy(raw: Optional[Dict[str, List[str]]]) -> Dict[str, Set[str]]:
    """
    Normalize taxonomy to set-based {canonical -> set(synonyms)}.
    """
    if not raw:
        return {}
    norm = {}
    for k, v in raw.items():
        base = k.strip().lower()
        syns = {base}
        for s in v or []:
            syns.add(str(s).strip().lower())
        norm[base] = syns
    return norm

def extract_keywords_from_jd(jd_text: str, nlp=None) -> Tuple[Set[str], Set[str], Dict[str, float], Optional[str]]:
    """
    Heuristic extraction from a freeform JD:
    - REQUIRED skills: lines containing 'required', 'must', 'minimum', 'mandatory'
    - NICE-TO-HAVE: lines with 'good to have', 'preferred', 'plus'
    - YEARS: parse 'X+ years' near a skill; fallback to overall 'X years'
    - Seniority hint: 'senior', 'lead', 'principal', etc.
    """
    lines = [l.strip() for l in jd_text.splitlines() if l.strip()]
    req, nice = set(), set()
    per_skill_req: Dict[str, float] = {}

    seniority_hint = None
    lower = jd_text.lower()
    if any(w in lower for w in ["principal", "staff"]):
        seniority_hint = "principal"
    elif "lead" in lower:
        seniority_hint = "lead"
    elif "senior" in lower:
        seniority_hint = "senior"
    elif "junior" in lower or "entry" in lower:
        seniority_hint = "junior"

    def collect_skills(text: str) -> Set[str]:
        out = set()
        # pull tech-like tokens
        for m in TECHLIKE.finditer(text):
            out.add(m.group(0).lower())
        # also capture UPPER or CapitalCase tokens in lists
        for term in re.findall(r"\b[A-Z][A-Za-z0-9+#.-]{2,}\b", text):
            t = term.lower()
            if len(t) <= 30 and not t.isdigit():
                out.add(t)
        return out

    for l in lines:
        ll = l.lower()
        items = collect_skills(l)
        if any(w in ll for w in ["required", "must", "minimum", "mandatory"]):
            req |= items
        elif any(w in ll for w in ["nice to have", "nice-to-have", "preferred", "good to have", "plus"]):
            nice |= items
        else:
            # generic bullet: bucket by presence of soft hints
            if items:
                if any(w in ll for w in ["strong", "proven", "expertise"]):
                    req |= items
                else:
                    nice |= items

        # per-skill years extraction like "3+ years with Java"
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\b.{0,40}\b([A-Za-z0-9+#.-]{2,})", l):
            years = float(m.group(1))
            skill = m.group(2).lower()
            per_skill_req[skill] = max(per_skill_req.get(skill, 0.0), years)

    # fallback: if req still empty, seed with tech-like from whole JD
    if not req:
        req = collect_skills(jd_text)

    # Clean overlaps
    nice -= req
    return req, nice, per_skill_req, seniority_hint

def compute_keyword_coverage(
    resume_text: str,
    res_tokens: List[str],
    required: Set[str],
    nice_to_have: Set[str],
    taxonomy: Dict[str, Set[str]],
):
    """
    Count presence & frequency, aggregate coverage, and produce rankings.
    """
    import re
    token_counts = collections.Counter(res_tokens)
    resume_lc = resume_text.lower()

    def synonyms_for(term: str) -> Set[str]:
        base = term.lower().strip()
        return taxonomy.get(base, {base})

    def count_for(term: str) -> Tuple[int, List[str]]:
        total = 0
        hits = []
        for alt in synonyms_for(term):
            alt = alt.lower().strip()
            total += token_counts.get(alt, 0)
            if " " in alt:
                total += len(re.findall(rf"(?<![A-Za-z0-9+#-]){re.escape(alt)}(?![A-Za-z0-9+#-])", resume_lc))
            if token_counts.get(alt, 0) > 0 or (" " in alt and alt in resume_lc):
                hits.append(alt)
        return int(total), hits

    details = []
    for t in sorted(required):
        c, syn = count_for(t)
        details.append({"term": t, "type": "required", "present": bool(c), "count": c, "weight": 2, "matched_synonyms": syn})
    for t in sorted(nice_to_have):
        c, syn = count_for(t)
        details.append({"term": t, "type": "nice", "present": bool(c), "count": c, "weight": 1, "matched_synonyms": syn})

    req_present = sum(1 for d in details if d["type"] == "required" and d["present"])
    nice_present = sum(1 for d in details if d["type"] == "nice" and d["present"])

    summary = {
        "required": {"needed": len(required), "present": req_present, "coverage": (req_present / max(1, len(required))) if required else 1.0},
        "nice": {"needed": len(nice_to_have), "present": nice_present, "coverage": (nice_present / max(1, len(nice_to_have))) if nice_to_have else 0.0},
    }

    missing_ranked = sorted([d for d in details if not d["present"]], key=lambda x: (-x["weight"], x["term"]))
    present_ranked = sorted([d for d in details if d["present"]], key=lambda x: (-x["weight"], -x["count"], x["term"]))

    return summary, details, missing_ranked, present_ranked
