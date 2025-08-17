from __future__ import annotations
import re
from typing import Dict, Set

YEARS_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\b", re.IGNORECASE)

def estimate_per_skill_years(resume_text: str, skills: Set[str]) -> Dict[str, float]:
    """
    Very lightweight heuristic:
    - For each skill occurrence, look +/- 80 chars for a 'X years' pattern.
    - Aggregate using max per skill (conservative).
    """
    text = resume_text
    low = text.lower()
    out: Dict[str, float] = {}

    for skill in skills:
        s = skill.lower()
        idx = 0
        best = 0.0
        while True:
            i = low.find(s, idx)
            if i == -1:
                break
            left = max(0, i - 80)
            right = min(len(text), i + len(s) + 80)
            window = text[left:right]
            for m in YEARS_PAT.finditer(window):
                years = float(m.group(1))
                if years > best:
                    best = years
            idx = i + len(s)
        if best > 0:
            out[skill] = round(best, 2)
    return out

def estimate_total_years(resume_text: str) -> float:
    """
    Pick the largest 'X years' reference as total experience proxy.
    """
    best = 0.0
    for m in YEARS_PAT.finditer(resume_text):
        yrs = float(m.group(1))
        if yrs > best:
            best = yrs
    return round(best, 2)
