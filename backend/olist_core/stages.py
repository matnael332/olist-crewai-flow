"""Pure stage runners ("run-then-narrate").

Each stage computes and writes every artifact deterministically, with template narratives. The Flow
calls these in both modes; in LLM mode it then kicks off the crew and calls the matching assemble_*
function to re-render the narrative files with the agents' narratives. Gates therefore never depend
on LLM behaviour.
"""

from __future__ import annotations

from pathlib import Path

from olist_core.cleaning.build import clean_olist
from olist_core.cleaning.geo import LOOKUP_FILE, load_centroids
from olist_core.columns import read_clean_data
from olist_core.contract.builder import build_contract, merge_narrative
from olist_core.eda.figures import make_eda_figures
from olist_core.eda.summary import compute_eda_summary
from olist_core.features.build import build_features
from olist_core.io import read_json, sha256_file, write_csv_stable, write_json_lf, write_text_lf
from olist_core.modeling.train import train_and_evaluate
from olist_core.narratives import (
    CleaningNarrative,
    ContractNarrative,
    EvaluationNarrative,
    FeatureNarrative,
    Insights,
    ModelCardNarrative,
)
from olist_core.policy import SEED
from olist_core.raw import load_raw_schema, load_raw_tables, profile_raw_tables
from olist_core.reporting import templates, templates_ds
from olist_core.reporting.html import render_eda_report_html
from olist_core.reporting.markdown import render_cleaning_md, render_insights_md
from olist_core.reporting.markdown_ds import render_evaluation_md, render_model_card_md

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


def _provenance(raw_dir: Path, lookups_dir: Path) -> dict:
    manifest = read_json(raw_dir / "manifest.json")
    return {
        "source": manifest.get("source"),
        "github_commit": manifest.get("github_commit"),
        "raw_files": {
            name: entry["sha256"] for name, entry in sorted(manifest.get("files", {}).items())
        },
        "zip_centroids_sha256": sha256_file(lookups_dir / LOOKUP_FILE),
    }


def run_analyst_stage(raw_dir: str | Path, lookups_dir: str | Path, out_dir: str | Path) -> dict:
    """Ingest, profile, clean, EDA and contract with template narratives. Returns a summary."""
    raw_dir, lookups_dir, out_dir = Path(raw_dir), Path(lookups_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = load_raw_tables(raw_dir)
    write_json_lf(
        out_dir / "raw_profile.json", profile_raw_tables(tables, load_raw_schema(raw_dir))
    )

    clean_df, cleaning_report = clean_olist(tables, load_centroids(lookups_dir))
    write_csv_stable(clean_df, out_dir / "clean_data.csv")
    write_json_lf(out_dir / "cleaning_report.json", cleaning_report)

    clean = read_clean_data(out_dir / "clean_data.csv")  # round trip: what downstream will read
    summary = compute_eda_summary(tables["orders"], clean, cleaning_report, figures=[])
    summary["figures"] = make_eda_figures(summary, clean, out_dir / "figures")
    write_json_lf(out_dir / "eda_summary.json", summary)

    write_json_lf(out_dir / "provenance.json", _provenance(raw_dir, lookups_dir))
    assemble_analyst_narratives(out_dir)
    return {
        "rows_in": cleaning_report["rows_in"],
        "rows_out": cleaning_report["rows_out"],
        "late_rate": summary["delivered_population"]["late_rate"],
        "figures": len(summary["figures"]),
    }


def assemble_analyst_narratives(
    out_dir: str | Path,
    cleaning: CleaningNarrative | None = None,
    insights: Insights | None = None,
    contract: ContractNarrative | None = None,
) -> dict[str, str]:
    """(Re)write the narrative artifacts. None means: use the deterministic template."""
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
    cleaning = cleaning or templates.cleaning_template(report)
    insights = insights or templates.insights_template(summary)
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
    narrative = merge_narrative(templates.contract_template(summary, report), contract)
    contract_doc = build_contract(out_dir / "clean_data.csv", report, provenance, narrative)
    write_json_lf(out_dir / "dataset_contract.json", contract_doc)
    return sources


def run_feature_stage(out_dir: str | Path) -> dict:
    """Re-validate the handoff, build features.csv and feature_spec.json (template narrative)."""
    out_dir = Path(out_dir)
    summary = build_features(out_dir / "clean_data.csv", out_dir / "dataset_contract.json", out_dir)
    assemble_feature_narrative(out_dir)
    return summary


def assemble_feature_narrative(
    out_dir: str | Path, narrative: FeatureNarrative | None = None
) -> str:
    out_dir = Path(out_dir)
    spec = read_json(out_dir / "feature_spec.json")
    contract = read_json(out_dir / "dataset_contract.json")
    source = "llm" if narrative else "template"
    narrative = narrative or templates_ds.feature_template(spec, contract)
    spec["narrative"] = {"narrative_source": source, **narrative.model_dump()}
    write_json_lf(out_dir / "feature_spec.json", spec)
    return source


def run_modeling_stage(out_dir: str | Path, seed: int = SEED) -> dict:
    """Train and compare models, write model.joblib, metrics and the two reports (templates)."""
    out_dir = Path(out_dir)
    summary = train_and_evaluate(
        out_dir / "features.csv",
        out_dir / "feature_spec.json",
        out_dir / "dataset_contract.json",
        out_dir,
        seed=seed,
    )
    assemble_modeling_narratives(out_dir)
    return summary


def assemble_modeling_narratives(
    out_dir: str | Path,
    evaluation: EvaluationNarrative | None = None,
    model_card: ModelCardNarrative | None = None,
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
    evaluation = evaluation or templates_ds.evaluation_template(metrics, meta)
    write_text_lf(
        out_dir / "evaluation_report.md",
        render_evaluation_md(metrics, meta, evaluation, sources["evaluation_report.md"], grounding),
    )
    card_template = templates_ds.model_card_template(metrics, meta)
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
    return sources
