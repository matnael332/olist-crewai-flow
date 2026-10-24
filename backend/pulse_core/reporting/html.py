"""eda_report.html: the key numbers and every EDA figure, self-contained (figures as base64)."""

from __future__ import annotations

import base64
import html
from pathlib import Path

from pulse_core.reporting.fmt import key_number

CSS = """
body { font-family: Inter, system-ui, sans-serif; margin: 2rem auto; max-width: 980px; color: #0f172a; }
h1 { font-size: 1.6rem; } h2 { font-size: 1.2rem; margin-top: 2rem; }
table { border-collapse: collapse; } td, th { padding: .35rem .8rem; border-bottom: 1px solid #e2e8f0; text-align: left; }
figure { margin: 1.5rem 0; } img { max-width: 100%; height: auto; } figcaption { color: #64748b; font-size: .9rem; }
.headline { background: #eff6ff; padding: 1rem; border-radius: .5rem; }
"""


def _figure(figures_dir: Path, name: str) -> str:
    path = figures_dir / name
    if not path.exists():
        return f"<figure><figcaption>{html.escape(name)} (missing)</figcaption></figure>"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(name)}">'
        f"<figcaption>{html.escape(name)}</figcaption></figure>"
    )


def render_eda_report_html(
    summary: dict, report: dict, figures_dir: str | Path, headline: str | None = None
) -> str:
    figures_dir = Path(figures_dir)
    rows = "".join(
        f"<tr><td>{html.escape(str(k['label']))}</td>"
        f"<td>{html.escape(key_number(str(k['label']), k['value']))}</td></tr>"
        for k in summary.get("key_numbers", [])
    )
    rules = "".join(
        f"<tr><td>{html.escape(r['rule'])}</td><td>{html.escape(r['description'])}</td>"
        f"<td>{r['rows_removed']:,}</td></tr>"
        for r in report.get("rules", [])
    )
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Exploratory data analysis</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>Exploratory data analysis</h1>",
        f"<p>Grain: {html.escape(str(summary.get('grain', '')))}</p>",
    ]
    if headline:
        parts.append(f"<p class='headline'>{html.escape(headline)}</p>")
    parts += [
        "<h2>Key numbers</h2><table><tr><th>metric</th><th>value</th></tr>",
        rows,
        "</table>",
        "<h2>Cleaning</h2><table><tr><th>rule</th><th>description</th><th>rows removed</th></tr>",
        rules,
        "</table>",
        "<h2>Figures</h2>",
        *[_figure(figures_dir, name) for name in summary.get("figures", [])],
        "</body></html>",
    ]
    return "\n".join(parts) + "\n"
