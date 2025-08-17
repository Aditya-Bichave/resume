from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from .html_templates import render_html_page

def build_report(
    overall_score: float,
    component_scores: Dict[str, float],
    formatting_penalty_pct: float,
    gates: Dict[str, bool],
    resume_text: str,
    jd_text: str,
    required,
    nice_to_have,
    per_skill_years_est: Dict[str,float],
    per_skill_years_req: Dict[str,float],
    years_total_detected: float,
    kw_summary: Dict[str, Any],
    kw_details: List[Dict[str, Any]],
    kw_missing_ranked: List[str],
    kw_present_ranked: List[Dict[str, Any]],
    suggestions: List[str],
    seniority_hint: Optional[str],
    diagnostics: Dict[str, Any],
):
    return {
        "overall_score": round(overall_score, 1),
        "component_scores": component_scores,
        "formatting_penalty_pct": formatting_penalty_pct,
        "gates": gates,
        "role_name": seniority_hint,
        "required_skills": sorted(list(required)),
        "nice_to_have_skills": sorted(list(nice_to_have)),
        "per_skill_years_required": {k: round(v,2) for k,v in per_skill_years_req.items()},
        "per_skill_years_estimated": {k: round(v,2) for k,v in per_skill_years_est.items()},
        "years_total_detected": years_total_detected,
        "keywords_summary": kw_summary,
        "keywords_details": kw_details,
        "keywords_missing_ranked": kw_missing_ranked,
        "keywords_present_ranked": kw_present_ranked,
        "suggestions": suggestions,
        "diagnostics": diagnostics,
    }

def write_json(report: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

def write_md(report: Dict[str, Any], path: Path) -> None:
    c = []
    c.append(f"# ATS Pro Plus\n\n**Overall Score:** **{report['overall_score']} / 100**\n")
    cs = report["component_scores"]
    c.append("## Component Scores\n")
    c.append("| Keywords | Semantic | Sections | Experience | Impact | Quality | Seniority | Format –% |\n")
    c.append("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    c.append(f"| {cs['keywords']} | {cs['semantic']} | {cs['sections']} | {cs['experience']} | {cs['impact']} | {cs['quality']} | {cs['seniority']} | -{cs['formatting_penalty_pct']}% |\n")

    c.append("\n## Gates\n")
    for k, v in report["gates"].items():
        c.append(f"- {'✅' if v else '❌'} {k.replace('_',' ').title()}")

    c.append("\n## Keywords Coverage\n")
    ks = report["keywords_summary"]
    c.append(f"- Required: {ks['required']['present']}/{ks['required']['needed']} ({int(ks['required']['coverage']*100)}%)")
    c.append(f"- Nice-to-have: {ks['nice']['present']}/{ks['nice']['needed']} ({int(ks['nice']['coverage']*100)}%)")

    miss = report["keywords_missing_ranked"][:10]
    if miss:
        c.append("\n**Top Missing:** " + ", ".join(miss))
    present = report["keywords_present_ranked"][:10]
    if present:
        c.append("\n**Top Present:** " + ", ".join(f"{d['term']} (x{d['count']})" for d in present))

    c.append("\n## Per-skill Years (Required vs Estimated)\n")
    req = report["per_skill_years_required"]; est = report["per_skill_years_estimated"]
    if req:
        c.append("| Skill | Required | Estimated | Coverage |\n|---|---:|---:|---:|\n")
        for k, v in sorted(req.items()):
            e = est.get(k, 0.0)
            cov = int(min(1.0, e/max(0.1,v)) * 100)
            c.append(f"| {k} | {v:.1f}y | {e:.1f}y | {cov}% |")
    else:
        c.append("_No per-skill years in JD._")

    c.append("\n\n## Suggestions\n")
    for s in report["suggestions"]:
        c.append(f"- {s}")

    path.write_text("\n".join(c), encoding="utf-8")

def write_html(report: Dict[str, Any], path: Path) -> None:
    html = render_html_page(report)
    path.write_text(html, encoding="utf-8")
