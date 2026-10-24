"""Markdown renderers. Tables of numbers always come from code; prose comes from a narrative
(template or LLM) and every file ends with a footer naming the narrative source and the hash of the
JSON the numbers came from (grounding)."""

from __future__ import annotations

from pulse_core.narratives import CleaningNarrative, Insights
from pulse_core.reporting.fmt import brl, bullets, md_table, num, pct


def footer(source: str, grounding: dict[str, str]) -> str:
    refs = ", ".join(f"{name} sha256={digest[:12]}" for name, digest in sorted(grounding.items()))
    return f"\n---\n<sub>narrative_source: {source} | grounding: {refs}</sub>\n"


def render_cleaning_md(
    report: dict, narrative: CleaningNarrative, source: str, grounding: dict[str, str]
) -> str:
    rows = [
        [
            r["rule"],
            r["description"],
            num(r["rows_before"]),
            num(r["rows_after"]),
            num(r["rows_removed"]),
        ]
        for r in report["rules"]
    ]
    recon = report["payment_reconciliation"]
    defs = report["definitions"]
    parts = [
        "# Cleaning report",
        "",
        narrative.summary,
        "",
        "## Rules applied (computed)",
        "",
        md_table(["rule", "description", "rows before", "rows after", "removed"], rows),
        "",
        "## Issues found",
        "",
        bullets(narrative.issues_found),
        "",
        "## Decisions",
        "",
        bullets(narrative.decisions),
        "",
        "## Payment reconciliation (computed)",
        "",
        md_table(
            ["check", "orders"],
            [
                ["payment_total within 1 cent of price + freight", num(recon["within_1_cent"])],
                [
                    "paid more (installment interest)",
                    num(recon["paid_more_than_price_plus_freight"]),
                ],
                ["paid less (vouchers)", num(recon["paid_less_than_price_plus_freight"])],
            ],
        ),
        "",
        "## Definitions",
        "",
        bullets([f"**{k}**: {v}" for k, v in sorted(defs.items())]),
        "",
        "## Caveats",
        "",
        bullets(narrative.caveats),
    ]
    return "\n".join(parts) + "\n" + footer(source, grounding)


def key_numbers_table(summary: dict) -> str:
    ao, dp = summary["all_orders"], summary["delivered_population"]
    rv, dl = dp["reviews"], dp["delivery"]
    rows = [
        ["All orders", "orders (every status)", num(ao["n_orders"])],
        ["All orders", "never delivered", pct(ao["non_delivered_share"])],
        [
            "All orders",
            "peak month",
            f"{ao['peak_month']['month']} ({num(ao['peak_month']['orders'])})",
        ],
        [
            "All orders",
            "busiest day",
            f"{ao['peak_day']['date']} ({num(ao['peak_day']['orders'])})",
        ],
        ["Delivered", "orders", num(dp["n_orders"])],
        [
            "Delivered",
            "customers (repeat share)",
            f"{num(dp['n_customers'])} ({pct(dp['repeat_customer_share'])})",
        ],
        ["Delivered", "GMV / AOV", f"{brl(dp['gmv_brl'])} / {brl(dp['aov_brl'])}"],
        ["Delivered", "late rate", f"{pct(dp['late_rate'])} ({num(dp['late_orders'])} orders)"],
        [
            "Delivered",
            "median delivery / promised days",
            f"{num(dl['median_days'])} / {num(dl['median_promised_days'])}",
        ],
        [
            "Delivered",
            "1-2 star reviews: late vs on time",
            f"{pct(rv['bad_review_rate_late'])} vs {pct(rv['bad_review_rate_on_time'])}",
        ],
    ]
    return md_table(["layer", "metric", "value"], rows)


def render_insights_md(
    summary: dict, narrative: Insights, source: str, grounding: dict[str, str]
) -> str:
    findings = []
    for i, f in enumerate(narrative.key_findings, start=1):
        findings += [
            f"### {i}. {f.title}",
            "",
            f"- **Evidence:** {f.evidence}",
            f"- **What it means:** {f.business_meaning}",
            "",
        ]
    figures = [f"- `figures/{name}`" for name in summary.get("figures", [])]
    parts = [
        "# Business insights: what happened",
        "",
        "Two layers: **all orders** (every status, business volume) and **delivered orders** "
        "(the clean population where lateness and reviews are measured).",
        "",
        "## Key numbers (computed)",
        "",
        key_numbers_table(summary),
        "",
        "## Headline",
        "",
        narrative.headline,
        "",
        "## Key findings",
        "",
        *findings,
        "## Recommendations",
        "",
        bullets(narrative.recommendations),
        "",
        "## Data caveats",
        "",
        bullets(narrative.data_caveats),
        "",
        "## Implications for modeling",
        "",
        bullets(narrative.modeling_implications),
        "",
        "## Figures",
        "",
        "\n".join(figures),
    ]
    return "\n".join(parts) + "\n" + footer(source, grounding)
