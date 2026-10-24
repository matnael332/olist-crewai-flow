"""Pure stage runners ("run-then-narrate").

Each stage computes and writes every artifact deterministically, with template narratives. The Flow
calls these in both modes; in LLM mode it then kicks off the crew and calls the matching assemble_*
function to re-render the narrative files with the agents' narratives. Gates therefore never depend
on LLM behaviour. The dataset spec supplies everything dataset-specific.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from pulse_core.columns import read_clean_data
from pulse_core.contract.builder import build_contract, merge_narrative
from pulse_core.features.build import build_features
from pulse_core.io import read_json, sha256_file, write_csv_stable, write_json_lf, write_text_lf
from pulse_core.modeling.train import train_and_evaluate
from pulse_core.narratives import (
    CleaningNarrative,
    ContractNarrative,
    EvaluationNarrative,
    FeatureNarrative,
    Insights,
    ModelCardNarrative,
)
from pulse_core.raw import load_raw_schema, load_raw_tables, profile_raw_tables
from pulse_core.reporting import templates, templates_ds
from pulse_core.reporting.html import render_eda_report_html
from pulse_core.reporting.markdown import render_cleaning_md, render_insights_md
from pulse_core.reporting.markdown_ds import render_evaluation_md, render_model_card_md
from pulse_core.scoring import SCORE_ROWS, score_current
from pulse_core.spec import DatasetSpec

NARRATIVES = "narratives.json"
ANALYST_ARTIFACTS = (
    "raw_profile.json",
    "clean_data.csv",
    "cleaning_report.json",
    "cleaning_report.md",
    "eda_summary.json",
    "eda_report.html",
    "insights.md",
    "dataset_contract.json",
)


def _provenance(spec: DatasetSpec, raw_dir: Path, lookups_dir: Path) -> dict:
    manifest = read_json(raw_dir / "manifest.json")
    return {
        "source": manifest.get("source") or spec.source,
        "raw_files": {
            name: entry["sha256"] for name, entry in sorted(manifest.get("files", {}).items())
        },
        "lookups": {name: sha256_file(lookups_dir / name) for name in spec.required_lookups},
    }


def record_narratives(out_dir: Path, language: str, **entries) -> None:
    """Merge narratives into narratives.json, the structured copy the website renders.

    Each entry is ``(narrative, source)``, the narrative a pydantic model or a dict; None removes
    the entry. A template narrative is in
    English; anything an agent wrote is in ``language``.
    """
    path = Path(out_dir) / NARRATIVES
    doc = read_json(path) if path.exists() else {}
    for name, entry in entries.items():
        if entry is None:
            doc.pop(name, None)
            continue
        narrative, source = entry
        doc[name] = {
            "source": source,
            "language": "en" if source == "template" else language,
            **(narrative.model_dump() if isinstance(narrative, BaseModel) else dict(narrative)),
        }
    write_json_lf(path, dict(sorted(doc.items())))


def run_analyst_stage(
    spec: DatasetSpec, raw_dir: str | Path, lookups_dir: str | Path, out_dir: str | Path
) -> dict:
    """Ingest, profile, clean, EDA and contract with template narratives. Returns a summary."""
    raw_dir, lookups_dir, out_dir = Path(raw_dir), Path(lookups_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = load_raw_tables(raw_dir)
    write_json_lf(
        out_dir / "raw_profile.json", profile_raw_tables(tables, load_raw_schema(raw_dir))
    )

    clean_df, cleaning_report = spec.clean(tables, lookups_dir)
    write_csv_stable(clean_df, out_dir / "clean_data.csv")
    write_json_lf(out_dir / "cleaning_report.json", cleaning_report)
    if spec.score_rows is not None:
        write_csv_stable(spec.score_rows(tables, lookups_dir), out_dir / SCORE_ROWS)

    clean = read_clean_data(out_dir / "clean_data.csv", spec.dtypes)  # what downstream reads
    summary = spec.eda_summary(tables, clean, cleaning_report)
    summary["grain"] = spec.grain
    summary["entity"] = spec.entity
    summary["figures"] = (
        spec.eda_figures(summary, clean, out_dir / "figures") if spec.eda_figures else []
    )
    write_json_lf(out_dir / "eda_summary.json", summary)

    write_json_lf(out_dir / "provenance.json", _provenance(spec, raw_dir, lookups_dir))
    assemble_analyst_narratives(spec, out_dir)
    return {
        "rows_in": cleaning_report["rows_in"],
        "rows_out": cleaning_report["rows_out"],
        "positive_rate": summary["positive_rate"],
        "figures": len(summary["figures"]),
    }


def assemble_analyst_narratives(
    spec: DatasetSpec,
    out_dir: str | Path,
    cleaning: CleaningNarrative | None = None,
    insights: Insights | None = None,
    contract: ContractNarrative | None = None,
    language: str = "en",
) -> dict[str, str]:
    """(Re)write the narrative artifacts. None means: use the deterministic template.

    ``language`` is the language the agents wrote in (templates are English)."""
    out_dir = Path(out_dir)
    report = read_json(out_dir / "cleaning_report.json")
    summary = read_json(out_dir / "eda_summary.json")
    provenance = read_json(out_dir / "provenance.json")
    grounding_clean = {"cleaning_report.json": sha256_file(out_dir / "cleaning_report.json")}
    grounding_eda = {"eda_summary.json": sha256_file(out_dir / "eda_summary.json")}

    sources = {
        "cleaning_report.md": "llm" if cleaning else "template",
        "insights.md": "llm" if insights else "template",
        "dataset_contract.json": "template+llm" if contract else "template",
    }
    cleaning = cleaning or (spec.cleaning_template or templates.cleaning_template)(spec, report)
    insights = insights or (spec.insights_template or templates.insights_template)(spec, summary)
    write_text_lf(
        out_dir / "cleaning_report.md",
        render_cleaning_md(report, cleaning, sources["cleaning_report.md"], grounding_clean),
    )
    write_text_lf(
        out_dir / "insights.md",
        render_insights_md(summary, insights, sources["insights.md"], grounding_eda),
    )
    write_text_lf(
        out_dir / "eda_report.html",
        render_eda_report_html(summary, report, out_dir / "figures", headline=insights.headline),
    )
    base = (spec.contract_template or templates.contract_template)(spec, summary, report)
    narrative = merge_narrative(base, contract, {c.name for c in spec.columns})
    contract_doc = build_contract(spec, out_dir / "clean_data.csv", report, provenance, narrative)
    write_json_lf(out_dir / "dataset_contract.json", contract_doc)
    record_narratives(
        out_dir,
        language,
        cleaning=(cleaning, sources["cleaning_report.md"]),
        insights=(insights, sources["insights.md"]),
        contract=(narrative, sources["dataset_contract.json"]),
    )
    return sources


def run_feature_stage(spec: DatasetSpec, out_dir: str | Path) -> dict:
    """Re-validate the handoff, build features.csv and feature_spec.json (template narrative)."""
    out_dir = Path(out_dir)
    summary = build_features(
        spec, out_dir / "clean_data.csv", out_dir / "dataset_contract.json", out_dir
    )
    assemble_feature_narrative(out_dir)
    return summary


def assemble_feature_narrative(
    out_dir: str | Path, narrative: FeatureNarrative | None = None, language: str = "en"
) -> str:
    out_dir = Path(out_dir)
    spec = read_json(out_dir / "feature_spec.json")
    contract = read_json(out_dir / "dataset_contract.json")
    source = "llm" if narrative else "template"
    narrative = narrative or templates_ds.feature_template(spec, contract)
    spec["narrative"] = {"narrative_source": source, **narrative.model_dump()}
    write_json_lf(out_dir / "feature_spec.json", spec)
    record_narratives(out_dir, language, features=(narrative, source))
    return source


def run_modeling_stage(spec: DatasetSpec, out_dir: str | Path, seed: int | None = None) -> dict:
    """Train and compare models, write model.joblib, metrics and the two reports (templates),
    then score the snapshot rows when the spec has them (scores_current.csv, reasons.json)."""
    out_dir = Path(out_dir)
    summary = train_and_evaluate(
        out_dir / "features.csv",
        out_dir / "feature_spec.json",
        out_dir / "dataset_contract.json",
        out_dir,
        seed=spec.seed if seed is None else seed,
    )
    summary["scoring"] = score_current(spec, out_dir)
    assemble_modeling_narratives(spec, out_dir)
    return summary


def assemble_modeling_narratives(
    spec: DatasetSpec,
    out_dir: str | Path,
    evaluation: EvaluationNarrative | None = None,
    model_card: ModelCardNarrative | None = None,
    language: str = "en",
) -> dict[str, str]:
    out_dir = Path(out_dir)
    metrics = read_json(out_dir / "metrics.json")
    meta = read_json(out_dir / "model_meta.json")
    contract = read_json(out_dir / "dataset_contract.json")
    grounding = {
        "metrics.json": sha256_file(out_dir / "metrics.json"),
        "model_meta.json": sha256_file(out_dir / "model_meta.json"),
    }
    sources = {
        "evaluation_report.md": "llm" if evaluation else "template",
        "model_card.md": "template+llm" if model_card else "template",
    }
    evaluation = evaluation or templates_ds.evaluation_template(metrics, meta, contract)
    write_text_lf(
        out_dir / "evaluation_report.md",
        render_evaluation_md(
            metrics, meta, contract, evaluation, sources["evaluation_report.md"], grounding
        ),
    )
    card_template = templates_ds.model_card_template(metrics, meta, contract, spec.model_card_items)
    write_text_lf(
        out_dir / "model_card.md",
        render_model_card_md(
            metrics,
            meta,
            contract,
            model_card or card_template,
            card_template,
            sources["model_card.md"],
            grounding,
        ),
    )
    record_narratives(
        out_dir,
        language,
        evaluation=(evaluation, sources["evaluation_report.md"]),
        model_card_general=(card_template, "template"),
        model_card=(model_card, "llm") if model_card else None,
    )
    return sources
