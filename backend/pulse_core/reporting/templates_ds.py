"""Deterministic baseline narratives for the Data Scientist crew artifacts (metrics-driven)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pulse_core.narratives import EvaluationNarrative, FeatureNarrative, ModelCardNarrative
from pulse_core.reporting.fmt import pct


def _label(contract: dict) -> str:
    return contract["target"].get("label") or contract["target"]["name"]


def _entity(contract: dict) -> str:
    return contract["target"].get("entity") or "row"


def feature_template(spec: dict, contract: dict) -> FeatureNarrative:
    validation = spec["input_validation"]
    rates = spec["target"]["positive_rate"]
    label = _label(contract)
    stage = contract["feature_rules"]["feature_available_at"]
    numeric = [f["name"] for f in spec["features"] if f["kind"] == "numeric"]
    categorical = [f["name"] for f in spec["features"] if f["kind"] == "categorical"]
    groups = []
    if numeric:
        groups.append(f"Numeric features: {', '.join(numeric)}.")
    if categorical:
        groups.append(f"Categorical features: {', '.join(categorical)}.")
    return FeatureNarrative(
        contract_compliance=(
            f"Follows dataset contract v{spec['contract_version']} (sha256 "
            f"{spec['contract_sha256'][:12]}): the handoff was re-validated before feature work "
            f"({validation['summary']}), {contract['row_count']:,} rows, all "
            f"{len(contract['required_features'])} required features built, none of the "
            f"{len(contract['forbidden_features'])} forbidden columns used."
        ),
        leakage_statement=(
            f"Every feature is derived from columns available at {stage}; "
            f"{', '.join(contract['forbidden_features'])} are excluded, so the model only sees "
            "what is known at decision time."
        ),
        feature_groups=groups or ["No features declared."],
        concerns=[
            f"The {label} rate differs between train ({pct(rates.get('train'))}) and test "
            f"({pct(rates.get('test'))}); thresholds must be re-tuned on recent data.",
        ],
    )


def _top_features(metrics: dict, k: int = 3) -> str:
    return ", ".join(f["feature"] for f in metrics["permutation_importance"]["features"][:k])


def evaluation_template(metrics: dict, meta: dict, contract: dict) -> EvaluationNarrative:
    chosen = metrics["selected_model"]
    sel = metrics["candidates"][chosen]
    test = sel["test"]
    base = metrics["candidates"]["baseline_prior"]["test"]
    label, entity = _label(contract), _entity(contract)
    others = [n for n in metrics["candidates"] if n not in (chosen, "baseline_prior")]
    comparison = f"{chosen} reached mean CV PR-AUC {sel['cv']['pr_auc_mean']:.3f}"
    if others:
        other = metrics["candidates"][others[0]]
        comparison += f" versus {other['cv']['pr_auc_mean']:.3f} for {others[0]}"
    comparison += (
        f"; on the test window its ROC-AUC is {test['roc_auc']:.3f} and PR-AUC "
        f"{test['pr_auc']:.3f} against a {base['pr_auc']:.3f} baseline."
    )
    lift = test["lift_at_top10pct"]
    return EvaluationNarrative(
        executive_summary=(
            f"The selected model ({chosen}) ranks {label} {entity} rows above chance: the "
            f"riskiest 10% of test rows are {label} {pct(test['precision_at_top10pct'])} of the "
            f"time, {f'{lift:.1f}x' if lift is not None else 'n/a'} the "
            f"{pct(test['prevalence'])} test rate."
        ),
        comparison_summary=comparison,
        strengths=[
            "Evaluated on later dates than it was trained on (time-aware split).",
            f"Uses only decision-time information; strongest signals: {_top_features(metrics)}.",
            "Lean, fast and fully reproducible pipeline (fixed seed and thread count).",
        ],
        weaknesses=[
            f"At threshold {test['threshold']:.3f} precision is {pct(test['precision'])} with "
            f"recall {pct(test['recall'])}.",
            f"The {label} rate drifts: {pct(meta['split']['train']['prevalence'])} in train vs "
            f"{pct(meta['split']['test']['prevalence'])} in test.",
        ],
        threshold_recommendation=(
            f"Use the score to rank {entity} rows and act on the riskiest 10% (precision "
            f"{pct(test['precision_at_top10pct'])}); re-tune the {test['threshold']:.3f} "
            "threshold on the most recent window before any automated action."
        ),
        next_steps=[
            "Retrain on a schedule and monitor the positive rate and calibration.",
            "Add features that capture history as of the decision time.",
        ],
    )


DEFAULT_ITEMS: dict[str, list[str]] = {
    "out_of_scope": ["Any decision that penalizes a person or account without human review."],
    "limitations": [
        "Associations, not causes: the model does not explain why an outcome happens.",
        "The positive rate is non-stationary; the decision threshold must be re-tuned.",
    ],
    "ethical_considerations": [
        "Scores are for prioritization only; they must never be the sole basis of an adverse action.",
    ],
    "monitoring_recommendations": [
        "Track the positive rate, PR-AUC and calibration on every new window.",
    ],
}


def model_card_template(
    metrics: dict, meta: dict, contract: dict, items: Mapping[str, Sequence[str]] | None = None
) -> ModelCardNarrative:
    test = metrics["candidates"][metrics["selected_model"]]["test"]
    train = meta["split"]["train"]
    test_split = meta["split"]["test"]
    label, entity = _label(contract), _entity(contract)
    extra = items or {}

    def merged(key: str) -> list[str]:
        return list(DEFAULT_ITEMS[key]) + [
            x for x in extra.get(key, []) if x not in DEFAULT_ITEMS[key]
        ]

    return ModelCardNarrative(
        purpose=(
            f"Estimate, at {contract['target'].get('decision_time', 'decision time')}, the "
            f"probability that a {entity} is {label}: {contract['target']['definition']}"
        ),
        intended_users="The team that operates the pipeline and acts on the ranked list.",
        out_of_scope=merged("out_of_scope"),
        training_data_summary=(
            f"{train['rows']:,} {entity} rows from {train['dates'][0]}..{train['dates'][1]} "
            f"({pct(train['prevalence'])} {label}); tested on {test_split['rows']:,} rows from "
            f"{test_split['dates'][0]}..{test_split['dates'][1]} ({pct(test['prevalence'])} {label})."
        ),
        limitations=merged("limitations"),
        ethical_considerations=merged("ethical_considerations"),
        monitoring_recommendations=merged("monitoring_recommendations"),
    )
