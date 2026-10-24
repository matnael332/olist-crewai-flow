"""Generic deterministic narratives for the Data Analyst artifacts.

A dataset spec may override any of these; without an override the pipeline is still
spec-complete in deterministic mode, with prose built from the computed numbers only.
"""

from __future__ import annotations

from pulse_core.narratives import (
    CleaningNarrative,
    ColumnNote,
    ContractNarrative,
    Finding,
    Insights,
)
from pulse_core.reporting.fmt import key_number, num, pct


def cleaning_template(spec, report: dict) -> CleaningNarrative:
    removed = [r for r in report["rules"] if r.get("rows_removed", 0) > 0]
    return CleaningNarrative(
        summary=(
            f"The deterministic cleaning pipeline turned {num(report['rows_in'])} raw rows into "
            f"{num(report['rows_out'])} clean rows ({spec.grain}) by applying "
            f"{len(report['rules'])} rules."
        ),
        issues_found=[f"{r['rule']}: {num(r['rows_removed'])} rows removed" for r in removed]
        or ["No rows had to be removed."],
        decisions=[f"{r['rule']}: {r['description']}" for r in report["rules"]],
        caveats=[
            "Rules are deterministic and documented; anything they do not cover remains in the data.",
        ],
    )


def insights_template(spec, summary: dict) -> Insights:
    rate = summary.get("positive_rate")
    findings = [
        Finding(
            title=str(k["label"]).capitalize(),
            evidence=f"{k['label']}: {key_number(str(k['label']), k['value'])}",
            business_meaning="Computed from the clean data; see the dataset contract for definitions.",
        )
        for k in summary.get("key_numbers", [])[:6]
    ]
    return Insights(
        headline=(
            f"{pct(rate)} of {spec.entity} rows are {spec.target_label} ({spec.target_definition})."
            if rate is not None
            else f"The clean {spec.entity} table is ready for modeling."
        ),
        key_findings=findings,
        recommendations=[
            f"Use the {spec.target_label} rate per split as the baseline when judging a model.",
        ],
        data_caveats=[f"Snapshot date {spec.snapshot_date}; later outcomes are not observed."],
        modeling_implications=[
            f"Only columns known at {spec.decision_stage} may become features (leakage rule).",
        ],
    )


def contract_template(spec, summary: dict, report: dict) -> ContractNarrative:
    notes = [
        ColumnNote(name=c.name, note=f"{c.description} (known at {c.available_at}).")
        for c in spec.columns
        if c.available_at not in ("identifier",)
    ][:8]
    return ContractNarrative(
        assumptions=[
            f"Grain: {spec.grain}; one row per {spec.entity} as of {spec.decision_time}.",
            f"Target {spec.target}: {spec.target_definition}",
        ],
        business_rules=[
            f"Features may only use columns tagged available_at == '{spec.decision_stage}'.",
            f"Splits are time-based on {spec.time_column}: "
            + "; ".join(f"{k} {v[0]}..{v[1]}" for k, v in spec.splits.items())
            + ".",
        ],
        column_notes=notes,
        known_limitations=[
            f"Outcomes after the snapshot date ({spec.snapshot_date}) are unknown.",
        ],
    )
