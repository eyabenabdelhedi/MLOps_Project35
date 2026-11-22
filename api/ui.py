import html
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.http import HttpResponse

from .services import CAT, NUM

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Churn explanation</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
 form{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.8rem}
 label{display:flex;flex-direction:column;font-size:.9rem;gap:.2rem}
 input,select{padding:.4rem}
 button{grid-column:1/-1;padding:.7rem;font-size:1rem;cursor:pointer}
 table{border-collapse:collapse;margin-top:1rem;width:100%}
 td,th{border:1px solid #ccc;padding:.4rem;text-align:left}
 .up{color:#c0392b}.down{color:#2471a3}
 img{max-width:100%;margin-top:1rem}
</style></head><body>
<h1>Customer churn: SHAP explanation</h1>
<form id="f">__FIELDS__<button type="submit">Explain this customer</button></form>
<div id="out"></div>
<script>
document.getElementById("f").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const out = document.getElementById("out");
  out.textContent = "Computing...";
  const body = {};
  new FormData(ev.target).forEach((v, k) => { body[k] = v === "" ? null : v; });
  const res = await fetch("/api/explain/", {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const d = await res.json();
  if (!res.ok) { out.textContent = d.error || JSON.stringify(d); return; }
  const rows = d.top_drivers.map(t =>
    `<tr><td>${t.feature}</td><td>${t.value.toFixed(3)}</td>
     <td class="${t.shap_value > 0 ? "up" : "down"}">${t.shap_value.toFixed(4)}</td>
     <td>${t.direction}</td></tr>`).join("");
  out.innerHTML = `<h2>Churn probability: ${(d.churn_probability * 100).toFixed(1)}%
    <small>(model v${d.model_version})</small></h2>
    <img src="data:image/png;base64,${d.waterfall_png_base64}" alt="SHAP waterfall">
    <table><tr><th>Feature</th><th>Value (processed)</th><th>SHAP</th><th>Effect</th></tr>${rows}</table>`;
});
</script></body></html>"""


def explain_page(request):
    ref = pd.read_csv(Path(settings.BASE_DIR) / "data" / "reference.csv")
    parts = []
    for c in NUM:
        med = float(ref[c].median())
        parts.append(f'<label>{c}<input name="{c}" type="number" step="any" value="{med:g}"></label>')
    for c in CAT:
        opts = "".join(f"<option>{html.escape(str(v))}</option>"
                       for v in sorted(ref[c].dropna().unique()))
        parts.append(f'<label>{c}<select name="{c}">{opts}</select></label>')
    return HttpResponse(PAGE.replace("__FIELDS__", "".join(parts)))