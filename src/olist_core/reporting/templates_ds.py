"""Deterministic baseline narratives for the Data Scientist crew artifacts."""

from __future__ import annotations

from olist_core.narratives import EvaluationNarrative, FeatureNarrative, ModelCardNarrative
from olist_core.reporting.fmt import pct


def feature_template(spec: dict, contract: dict) -> FeatureNarrative:
    validation = spec["input_validation"]
    rates = spec["target"]["positive_rate"]
    return FeatureNarrative(
        contract_compliance=(
            f"Follows dataset contract v{spec['contract_version']} (sha256 "
            f"{spec['contract_sha256'][:12]}): the handoff was re-validated before feature work "
            f"({validation['summary']}), {contract['row_count']:,} rows, all "
            f"{len(contract['required_features'])} required features built, none of the "
            f"{len(contract['forbidden_features'])} forbidden columns used."
        ),
        leakage_statement=(
            "Every feature is derived from columns available at checkout; delivery timestamps, "
            "delivery_days and review fields are excluded, so the model only sees what is known "
            "when the customer pays."
        ),
        feature_groups=[
            "Basket: item, product and seller counts, price, freight and freight ratio.",
            "Package: total weight and volume and the largest dimensions.",
            "Payment: amount, installments, number of payments and payment type.",
            "Geography: customer and seller states, same-state flag, distance.",
            "Promise and timing: promised days, purchase month, weekday, hour, weekend flag.",
            "Product: category of the first item.",
        ],
        concerns=[
            f"The late rate differs between train ({pct(rates['train'])}) and test "
            f"({pct(rates['test'])}); thresholds must be re-tuned on recent data.",
            "Package dimensions and distance are missing for a small share of orders.",
        ],
    )


def _top_features(metrics: dict, k: int = 3) -> str:
    return ", ".join(f["feature"] for f in metrics["permutation_importance"]["features"][:k])


def evaluation_template(metrics: dict, meta: dict) -> EvaluationNarrative:
    chosen = metrics["selected_model"]
    sel = metrics["candidates"][chosen]
    test = sel["test"]
    base = metrics["candidates"]["baseline_prior"]["test"]
    others = [n for n in metrics["candidates"] if n not in (chosen, "baseline_prior")]
    comparison = f"{chosen} reached mean CV PR-AUC {sel['cv']['pr_auc_mean']:.3f}"
    if others:
        other = metrics["candidates"][others[0]]
        comparison += f" versus {other['cv']['pr_auc_mean']:.3f} for {others[0]}"
    comparison += (
        f"; on the test window its ROC-AUC is {test['roc_auc']:.3f} and PR-AUC "
        f"{test['pr_auc']:.3f} against a {base['pr_auc']:.3f} baseline."
    )
    return EvaluationNarrative(
        executive_summary=(
            f"The selected model ({chosen}) ranks late orders well above chance: the riskiest 10% "
            f"of test orders are late {pct(test['precision_at_top10pct'])} of the time, "
            f"{test['lift_at_top10pct']:.1f}x the {pct(test['prevalence'])} test late rate."
        ),
        comparison_summary=comparison,
        strengths=[
            "Evaluated on later months than it was trained on (time-aware split).",
            f"Uses only checkout-time information; strongest signals: {_top_features(metrics)}.",
            "Lean, fast and fully reproducible pipeline (fixed seed and thread count).",
        ],
        weaknesses=[
            f"Modest absolute precision: at threshold {test['threshold']:.3f} precision is "
            f"{pct(test['precision'])} with recall {pct(test['recall'])}.",
            f"The late rate drifts: {pct(meta['split']['train']['prevalence'])} in train vs "
            f"{pct(meta['split']['test']['prevalence'])} in test.",
            "Delivered orders only: orders that never arrived are not represented.",
        ],
        threshold_recommendation=(
            "Use the score to rank orders and act on the riskiest 10% (precision "
            f"{pct(test['precision_at_top10pct'])}); re-tune the {test['threshold']:.3f} "
            "threshold on the most recent months before any automated action."
        ),
        next_steps=[
            "Add seller history features (past late rate, as of the purchase time).",
            "Retrain monthly and monitor the late rate and calibration by state.",
            "Model delivery days (regression) to set better delivery promises.",
        ],
    )


def model_card_template(metrics: dict, meta: dict) -> ModelCardNarrative:
    test = metrics["candidates"][metrics["selected_model"]]["test"]
    train = meta["split"]["train"]
    test_split = meta["split"]["test"]
    return ModelCardNarrative(
        purpose=(
            "Estimate, at checkout, the probability that an Olist order will be delivered after "
            "the date promised to the customer, so operations can prioritize risky orders "
            "(proactive messages, carrier upgrades, promise adjustments)."
        ),
        intended_users="Marketplace operations and customer-experience teams.",
        out_of_scope=[
            "Denying service, changing prices or penalizing customers or sellers.",
            "Orders outside Brazil or from other marketplaces without revalidation.",
            "Predicting the exact delivery date (this is a yes/no lateness risk).",
        ],
        training_data_summary=(
            f"{train['rows']:,} delivered orders purchased {train['dates'][0]}..{train['dates'][1]} "
            f"({pct(train['prevalence'])} late); tested on {test_split['rows']:,} orders from "
            f"{test_split['dates'][0]}..{test_split['dates'][1]} ({pct(test['prevalence'])} late)."
        ),
        limitations=[
            "Measured among delivered orders only; orders that never arrived are excluded.",
            "Right-censoring: August 2018 is held out because late orders may not have arrived "
            "by the 2018-10-17 snapshot.",
            "The late rate is non-stationary; the decision threshold must be re-tuned on recent "
            "data.",
            "Data from 2017-2018 on one marketplace; patterns may have changed.",
            "Associations, not causes: the model does not explain why an order is late.",
        ],
        ethical_considerations=[
            "Regional bias: customers in North and Northeast states have higher late rates; "
            "interventions must improve their service, never deprioritize them.",
            "Seller fairness: scores are for operational prioritization only, not seller penalties.",
            "Privacy: identifiers are pseudonymous, but zip prefix plus timestamps are "
            "quasi-identifiers; only aggregates are shown in dashboards.",
            "License: the Olist data is shared for non-commercial use (CC BY-NC-SA 4.0 listing); "
            "this model is coursework, not a production system.",
            "Portuguese review text is never used, avoiding sensitive free-text content.",
        ],
        monitoring_recommendations=[
            "Track the monthly late rate, PR-AUC and calibration, overall and by state.",
            "Re-tune the threshold and retrain when the late rate shifts.",
            "Alert when input distributions (promised days, freight, states) drift.",
        ],
    )
