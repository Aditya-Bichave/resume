import argparse
from pathlib import Path
from .parser import extract_pdf_text, read_text_file
from .tokenize import get_nlp, basic_tokenize
from .skills import build_taxonomy, extract_keywords_from_jd, compute_keyword_coverage
from .experience import estimate_per_skill_years, estimate_total_years
from .scoring import compute_component_scores, compute_gates, build_suggestions
from .report import build_report, write_json, write_md, write_html
from .utils import load_json_if_exists, as_float

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ATS Pro Plus — resume vs JD analyzer")
    p.add_argument("--pdf", required=True, help="Path to resume PDF")
    p.add_argument("--jd", required=True, help="Path to job description text")
    p.add_argument("--taxonomy", help="JSON file mapping term -> [synonyms]")
    p.add_argument("--out-json", default="ats-pro-report.json")
    p.add_argument("--out-md", default=None)
    p.add_argument("--out-html", default=None)
    p.add_argument("--min-score", type=float, default=78.0)
    p.add_argument("--min-req-coverage", type=float, default=0.85)
    p.add_argument("--snippets", type=int, default=5, help="Semantic snippet count")
    p.add_argument("--grammar-check", action="store_true", help="Use language_tool if available")
    p.add_argument("--fast", action="store_true", help="Skip heavy checks/HTML")
    return p

def main() -> int:
    args = build_arg_parser().parse_args()

    pdf_path = Path(args.pdf)
    jd_path = Path(args.jd)
    taxonomy_path = Path(args.taxonomy) if args.taxonomy else None

    resume_text = extract_pdf_text(pdf_path)
    jd_text = read_text_file(jd_path)

    taxonomy = build_taxonomy(load_json_if_exists(taxonomy_path))

    # NLP/tokenization
    nlp = get_nlp()  # spaCy pipeline or None
    res_tokens, res_sents = basic_tokenize(resume_text, nlp)
    jd_tokens, jd_sents = basic_tokenize(jd_text, nlp)

    # Keyword sets from JD
    required, nice_to_have, min_years_map, seniority_hint = extract_keywords_from_jd(
        jd_text, nlp
    )

    # Keyword coverage & ranking (with synonyms)
    kw_summary, kw_details, kw_missing_ranked, kw_present_ranked = compute_keyword_coverage(
        resume_text, res_tokens, required, nice_to_have, taxonomy
    )

    # Per-skill years and totals
    per_skill_years_est = estimate_per_skill_years(resume_text, required | nice_to_have)
    years_total_detected = estimate_total_years(resume_text)

    # Component scores
    component_scores, diagnostics = compute_component_scores(
        resume_text=resume_text,
        jd_text=jd_text,
        res_tokens=res_tokens,
        jd_tokens=jd_tokens,
        res_sents=res_sents,
        nlp=nlp,
        per_skill_years_est=per_skill_years_est,
        per_skill_years_req=min_years_map,
        kw_summary=kw_summary,
        kw_details=kw_details,
        snippets=args.snippets,
        fast=args.fast,
        grammar_check=args.grammar_check,
        seniority_hint=seniority_hint,
    )
    overall_score = diagnostics["overall_score"]
    formatting_penalty_pct = diagnostics.get("formatting_penalty_pct", 0)

    # Gates
    gates = compute_gates(
        overall_score=overall_score,
        required_coverage=kw_summary["required"]["coverage"],
        per_skill_years_est=per_skill_years_est,
        per_skill_years_req=min_years_map,
        seniority_ok=diagnostics.get("seniority_ok", True),
        min_score=as_float(args.min_score, 78.0),
        min_req_cov=as_float(args.min_req_coverage, 0.85),
        anti_gaming_flags=diagnostics.get("anti_gaming", {}),
    )

    # Suggestions
    suggestions = build_suggestions(
        kw_missing_ranked=kw_missing_ranked,
        nice_missing_ranked=[d for d in kw_missing_ranked if d.get("type") == "nice"],
        lacking_years=diagnostics.get("lacking_years", {}),
        per_skill_years_est=per_skill_years_est,
        digits_ratio=diagnostics.get("digits_ratio", 0.0),
        readability_norm=diagnostics.get("readability_norm", 1.0),
        gates=gates,
        seniority_ok=diagnostics.get("seniority_ok", True),
    )

    report = build_report(
        overall_score=overall_score,
        component_scores=component_scores,
        formatting_penalty_pct=formatting_penalty_pct,
        gates=gates,
        resume_text=resume_text,
        jd_text=jd_text,
        required=required,
        nice_to_have=nice_to_have,
        per_skill_years_est=per_skill_years_est,
        per_skill_years_req=min_years_map,
        years_total_detected=years_total_detected,
        kw_summary=kw_summary,
        kw_details=kw_details,
        kw_missing_ranked=[d["term"] for d in kw_missing_ranked],
        kw_present_ranked=[{"term": d["term"], "count": d["count"], "type": d["type"]} for d in kw_present_ranked],
        suggestions=suggestions,
        seniority_hint=seniority_hint,
        diagnostics=diagnostics,
    )

    write_json(report, Path(args.out_json))

    if args.out_md and not args.fast:
        write_md(report, Path(args.out_md))

    if args.out_html and not args.fast:
        write_html(report, Path(args.out_html))

    return 0
