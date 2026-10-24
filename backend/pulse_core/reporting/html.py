"""Self-contained eda_report.html: inline CSS, base64 figures, no external requests."""

from __future__ import annotations

import base64
import html
from pathlib import Path

from pulse_core.reporting.fmt import brl, num, pct

CSS = """
body{font-family:Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#f8fafc;color:#0f172a}
main{max-width:1080px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:32px 0 12px;border-bottom:1px solid #e2e8f0;padding-bottom:6px}
p.meta{color:#475569;font-size:13px;margin:0 0 16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px}
.card .k{font-size:12px;color:#64748b}.card .v{font-size:20px;font-weight:600;margin-top:4px}
.layer{font-size:12px;font-weight:600;color:#2563eb;text-transform:uppercase;letter-spacing:.04em;margin:18px 0 8px}
table{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin:8px 0}
th,td{border:1px solid #e2e8f0;padding:6px 8px;text-align:left}th{background:#f1f5f9}
figure{margin:16px 0;background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:10px}
figure img{max-width:100%;height:auto}figcaption{font-size:12px;color:#475569;margin-top:6px}
.note{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:10px 12px;font-size:13px}
"""

CAPTIONS = {
    "monthly_orders_all_vs_delivered.png": "All orders per month, delivered vs never delivered.",
    "late_rate_by_month.png": "Late-delivery rate by purchase month (delivered orders).",
    "late_rate_by_state.png": "Late-delivery rate by customer state (delivered orders).",
    "delivery_days_hist.png": "Distribution of days from purchase to delivery.",
    "review_score_vs_late.png": "Review scores for late vs on-time orders.",
    "gmv_by_category_top10.png": "Top 10 categories by GMV (delivered orders).",
    "promised_vs_actual_by_month.png": "Median promised vs actual delivery days by month.",
}


def _card(key: str, value: str) -> str:
    return f'<div class="card"><div class="k">{html.escape(key)}</div><div class="v">{html.escape(value)}</div></div>'


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _figure(figures_dir: Path, name: str) -> str:
    data = base64.b64encode((figures_dir / name).read_bytes()).decode("ascii")
    caption = html.escape(CAPTIONS.get(name, name))
    return (
        f'<figure><img alt="{caption}" src="data:image/png;base64,{data}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def render_eda_report_html(
    summary: dict, cleaning_report: dict, figures_dir: str | Path, headline: str | None = None
) -> str:
    figures_dir = Path(figures_dir)
    ao, dp = summary["all_orders"], summary["delivered_population"]
    rv, dl = dp["reviews"], dp["delivery"]
    cards_all = "".join(
        [
            _card("orders (every status)", num(ao["n_orders"])),
            _card("never delivered", pct(ao["non_delivered_share"])),
            _card("peak month", f"{ao['peak_month']['month']}: {num(ao['peak_month']['orders'])}"),
            _card("busiest day", f"{ao['peak_day']['date']}: {num(ao['peak_day']['orders'])}"),
        ]
    )
    cards_delivered = "".join(
        [
            _card("delivered orders", num(dp["n_orders"])),
            _card("GMV", brl(dp["gmv_brl"])),
            _card("AOV", brl(dp["aov_brl"])),
            _card("late rate", pct(dp["late_rate"])),
            _card("median delivery days", num(dl["median_days"])),
            _card(
                "bad reviews: late / on time",
                f"{pct(rv['bad_review_rate_late'], 0)} / {pct(rv['bad_review_rate_on_time'], 0)}",
            ),
        ]
    )
    rules = _table(
        ["rule", "description", "rows removed"],
        [[r["rule"], r["description"], num(r["rows_removed"])] for r in cleaning_report["rules"]],
    )
    states = _table(
        ["state", "orders", "late rate", "median delivery days"],
        [
            [s["state"], num(s["orders"]), pct(s["late_rate"]), num(s["median_delivery_days"])]
            for s in dp["late_rate_by_state"]
        ],
    )
    cats = _table(
        ["category", "orders", "GMV", "late rate"],
        [
            [c["category"], num(c["orders"]), brl(c["gmv_brl"]), pct(c["late_rate"])]
            for c in dp["categories_top10_by_gmv"]
        ],
    )
    figs = {name: _figure(figures_dir, name) for name in summary.get("figures", [])}
    headline_html = (
        f'<p class="note"><b>Headline.</b> {html.escape(headline)}</p>' if headline else ""
    )
    sections = [
        "<h1>Olist e-commerce: exploratory data analysis</h1>",
        '<p class="meta">Source: Olist Brazilian E-Commerce Public Dataset (GitHub '
        "olist/work-at-olist-data). Generated by the Data Analyst crew's deterministic EDA step.</p>",
        headline_html,
        '<div class="layer">All orders (every status)</div>',
        f'<div class="cards">{cards_all}</div>',
        '<div class="layer">Delivered orders (clean population)</div>',
        f'<div class="cards">{cards_delivered}</div>',
        "<h2>1. Business volume and growth</h2>",
        figs.get("monthly_orders_all_vs_delivered.png", ""),
        "<h2>2. Delivery performance</h2>",
        figs.get("delivery_days_hist.png", ""),
        figs.get("promised_vs_actual_by_month.png", ""),
        figs.get("late_rate_by_month.png", ""),
        "<h2>3. Geography</h2>",
        figs.get("late_rate_by_state.png", ""),
        states,
        "<h2>4. Customer satisfaction</h2>",
        figs.get("review_score_vs_late.png", ""),
        "<h2>5. Products</h2>",
        figs.get("gmv_by_category_top10.png", ""),
        cats,
        "<h2>6. Data cleaning</h2>",
        f"<p>{num(cleaning_report['rows_in'])} raw orders became "
        f"{num(cleaning_report['rows_out'])} clean delivered orders.</p>",
        rules,
    ]
    body = "\n".join(s for s in sections if s)
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Olist EDA report</title>\n<style>{CSS}</style>\n</head>\n"
        f"<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n"
    )
