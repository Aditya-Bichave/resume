def render_html_page(r: dict) -> str:
    cs = r.get("component_scores", {})
    ks = r.get("keywords_summary", {"required": {"present":0,"needed":0,"coverage":0.0}, "nice":{"present":0,"needed":0,"coverage":0.0}})
    gates = r.get("gates", {})
    req = r.get("per_skill_years_required", {})
    est = r.get("per_skill_years_estimated", {})
    miss = r.get("keywords_missing_ranked", [])[:10]
    present = r.get("keywords_present_ranked", [])[:10]

    def bar(pct: int) -> str:
        pct = max(0, min(100, int(pct)))
        return f"<div class='bar'><span style='width:{pct}%;'></span></div> {pct}%"

    def row(label: str, val: float) -> str:
        return f"<tr><td>{label}</td><td>{bar(int(val))}</td></tr>"

    per_skill_rows = ""
    if req:
        for k, v in sorted(req.items()):
            e = est.get(k, 0.0)
            cov = int(min(1.0, e/max(0.1, v)) * 100)
            per_skill_rows += f"<tr><td>{k}</td><td>{v:.1f}y</td><td>{e:.1f}y</td><td>{bar(cov)}</td></tr>"
    else:
        per_skill_rows = "<tr><td colspan='4'>No per-skill year requirements provided in JD.</td></tr>"

    gate_items = "".join(
        f"<li class='{('ok' if v else 'bad')}'>{'✔' if v else '✖'} {k.replace('_',' ').title()}</li>"
        for k, v in gates.items()
    )

    miss_html = "<ul>" + "".join(f"<li><strong>{t}</strong></li>" for t in miss) + "</ul>" if miss else "<div>—</div>"
    present_html = "<ul>" + "".join(f"<li><strong>{d['term']}</strong> <span class='muted'>(x{d.get('count',0)})</span></li>" for d in present) + "</ul>" if present else "<div>—</div>"

    suggestions = r.get("suggestions", [])
    sug_html = "<ul>" + "".join(f"<li>{s}</li>" for s in suggestions) + "</ul>"

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>ATS Pro Plus Report</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
:root{{--bg:#0b0d10;--card:#11151b;--fg:#e7edf3;--muted:#9aa4b2;--accent:#4c8bf5;--ok:#19c37d;--bad:#ff5a5f;--bar:#1f2630;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",Arial}}
a{{color:var(--accent);text-decoration:none}} header{{padding:18px;border-bottom:1px solid #1c212b;background:linear-gradient(180deg,#0b0d10,#0c1117)}}
h1{{margin:0;font-size:18px}} main{{max-width:1000px;margin:0 auto;padding:18px;display:grid;gap:14px}}
.card{{background:var(--card);border:1px solid #1c212b;border-radius:12px;padding:14px}} .muted{{color:var(--muted)}}
.table{{width:100%;border-collapse:collapse}} .table th,.table td{{border-bottom:1px dashed #1f2633;padding:8px;text-align:left}}
.badge{{display:inline-flex;align-items:center;gap:6px;border:1px solid #253044;border-radius:999px;padding:4px 10px;color:#b9cff5}}
.row{{display:flex;align-items:center;justify-content:space-between;gap:12px}} .two{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
ul{{margin:8px 0 0 18px}} .bar{{height:8px;background:var(--bar);border-radius:999px;overflow:hidden;display:inline-block;min-width:140px;margin-right:8px}}
.bar>span{{display:block;height:100%;background:linear-gradient(90deg,#3d7bfd,#00c6ff)}} li.ok{{color:var(--ok)}} li.bad{{color:var(--bad)}}
</style>
</head>
<body>
  <header>
    <h1>ATS Pro Plus <span class="muted">· Report</span></h1>
  </header>
  <main>
    <section class="card">
      <div class="row">
        <div>
          <div class="muted">Overall Score</div>
          <div style="font-size:28px;font-weight:700">{r.get('overall_score','—')}</div>
        </div>
        <div class="badge">Formatting Penalty −{cs.get('formatting_penalty_pct', 0)}%</div>
      </div>
    </section>

    <section class="card">
      <h2>Component Scores</h2>
      <table class="table">
        <tr><th>Component</th><th>Score</th></tr>
        {row("Keywords", cs.get("keywords",0))}
        {row("Semantic", cs.get("semantic",0))}
        {row("Sections", cs.get("sections",0))}
        {row("Experience", cs.get("experience",0))}
        {row("Impact", cs.get("impact",0))}
        {row("Quality", cs.get("quality",0))}
        {row("Seniority", cs.get("seniority",0))}
      </table>
    </section>

    <section class="card">
      <h2>Gates</h2>
      <ul>
        {gate_items}
      </ul>
    </section>

    <section class="card">
      <h2>Keywords Coverage</h2>
      <table class="table">
        <tr><th></th><th>Present</th><th>Needed</th><th>Coverage</th></tr>
        <tr>
          <td>Required</td>
          <td>{ks['required']['present']}</td>
          <td>{ks['required']['needed']}</td>
          <td>{bar(int(ks['required']['coverage']*100))}</td>
        </tr>
        <tr>
          <td>Nice-to-have</td>
          <td>{ks['nice']['present']}</td>
          <td>{ks['nice']['needed']}</td>
          <td>{bar(int(ks['nice']['coverage']*100))}</td>
        </tr>
      </table>
      <div class="two">
        <div>
          <h3>Top Missing</h3>
          {miss_html}
        </div>
        <div>
          <h3>Top Present</h3>
          {present_html}
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Per-skill Years (Required vs Estimated)</h2>
      <table class="table">
        <tr><th>Skill</th><th>Required</th><th>Estimated</th><th>Coverage</th></tr>
        {per_skill_rows}
      </table>
    </section>

    <section class="card">
      <h2>Suggestions</h2>
      {sug_html}
    </section>
  </main>
</body>
</html>"""
