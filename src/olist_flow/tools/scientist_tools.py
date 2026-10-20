"""Data Scientist crew tools. (No `from __future__ import annotations`: @tool reads annotations.)"""

from crewai.tools import tool

from olist_core.io import read_json
from olist_core.validators.contract_gate import validate_clean_data_against_contract
from olist_flow.runctx import get_context
from olist_flow.tools.analyst_tools import _safe


def validation_view() -> dict:
    art = get_context().artifacts_dir
    report = validate_clean_data_against_contract(
        art / "clean_data.csv", art / "dataset_contract.json"
    )
    return {
        "passed": report.passed,
        "summary": report.summary,
        "failed_checks": report.errors,
        "checks": [c["name"] for c in report.checks],
    }


def features_view() -> dict:
    spec = read_json(get_context().artifacts_dir / "feature_spec.json")
    return {
        "input_validation": spec["input_validation"]["summary"],
        "contract_version": spec["contract_version"],
        "contract_sha256": spec["contract_sha256"][:12],
        "rows_by_split": spec["split"]["rows"],
        "late_rate_by_split": spec["target"]["positive_rate"],
        "model_features": [
            {"name": f["name"], "kind": f["kind"], "derived_from": f["derived_from"]}
            for f in spec["features"]
        ],
        "excluded_columns": [c["name"] for c in spec["excluded_columns"]],
        "leakage_rule": spec["leakage_rule"],
    }


def training_view() -> dict:
    art = get_context().artifacts_dir
    metrics = read_json(art / "metrics.json")
    meta = read_json(art / "model_meta.json")
    keep = (
        "roc_auc",
        "pr_auc",
        "precision_at_top10pct",
        "lift_at_top10pct",
        "prevalence",
        "threshold",
        "precision",
        "recall",
        "f1",
        "brier",
    )
    return {
        "selected_model": metrics["selected_model"],
        "selection": metrics["selection"],
        "cv_scheme": metrics["cv_scheme"],
        "candidates": {
            name: {
                "description": c["description"],
                "cv_pr_auc_mean": c["cv"]["pr_auc_mean"],
                "cv_roc_auc_mean": c["cv"]["roc_auc_mean"],
                "cv_fold_late_rates": [f["val_prevalence"] for f in c["cv"]["folds"]],
                "test": {k: c["test"][k] for k in keep},
            }
            for name, c in metrics["candidates"].items()
        },
        "split": metrics["split"],
        "censored_buffer": meta["censored_buffer"],
        "top_features": metrics["permutation_importance"]["features"][:8],
    }


@tool("validate_inputs")
def validate_inputs() -> str:
    """Re-validate the cleaned data handed over by the Data Analyst crew against the dataset
    contract (columns, dtypes, nulls, bounds, allowed values, patterns, constraints)."""
    return _safe(validation_view)


@tool("build_features", result_as_answer=True)
def build_features() -> str:
    """Return the feature spec built from checkout-time columns: validation result, rows and late
    rate per split, the model features with their source columns, and the excluded columns."""
    return _safe(features_view)


@tool("train_models", result_as_answer=True)
def train_models() -> str:
    """Return the model comparison: baseline, logistic regression and gradient boosting with
    time-aware cross-validation and test-window metrics, the selected model and top features."""
    return _safe(training_view)
