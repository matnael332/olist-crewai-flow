"""Markdown renderers. Tables of numbers always come from code; prose comes from a narrative
(template or LLM) and every file ends with a footer naming the narrative source and the hash of the
JSON the numbers came from (grounding)."""

from __future__ import annotations

from pulse_core.narratives import CleaningNarrative, Insights
from pulse_core.reporting.fmt import bullets, key_number, md_table, num


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
    defs = report.get("definitions") or {}
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
    rows = [
        [k["label"], key_number(k["label"], k["value"])] for k in summary.get("key_numbers", [])
    ]
    return md_table(["metric", "value"], rows)


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
        f"Grain: {summary.get('grain', 'see the dataset contract')}. Every number below is "
        "computed from the clean data; the prose comes from the narrative.",
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
        "\n".join(figures) if figures else "- (none)",
    ]
    return "\n".join(parts) + "\n" + footer(source, grounding)
