"""Markdown renderers for evaluation_report.md and model_card.md.

Tables and facts always come from metrics.json / model_meta.json / the contract (code). Prose comes
from the narrative. For the model card, limitation, ethics and monitoring bullets keep the template
items and append new LLM items, so the required sections are never empty."""

from __future__ import annotations

from pulse_core.narratives import EvaluationNarrative, ModelCardNarrative
from pulse_core.reporting.fmt import bullets, md_table, num, pct
from pulse_core.reporting.markdown import footer


def _f3(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _pm(mean, std) -> str:
    return "n/a" if mean is None else f"{mean:.3f} ± {std:.3f}"


MIN_EXTRA_WORDS = 5


def merge_items(base: list[str], extra: list[str] | None) -> list[str]:
    """Template items first, then LLM additions that are full statements and not repeats.

    Additions under five words are dropped: they are topic labels, not statements (a model once
    echoed its prompt's topic list back as list items).
    """
    seen = {x.strip().lower() for x in base}
    out = list(base)
    for item in extra or []:
        text = item.strip()
        if len(text.split()) >= MIN_EXTRA_WORDS and text.lower() not in seen:
            out.append(text)
            seen.add(text.lower())
    return out


SPLIT_ORDER = ("train", "test")


def _label(contract: dict) -> str:
    return contract["target"].get("label") or contract["target"]["name"]


def _entity(contract: dict) -> str:
    return contract["target"].get("entity") or "row"


def split_table(meta: dict, contract: dict) -> str:
    # Chronological order; the JSON keys are sorted alphabetically.
    names = [n for n in SPLIT_ORDER if n in meta["split"]]
    names += [n for n in meta["split"] if n not in names]
    rows = []
    for name in names:
        s = meta["split"][name]
        rows.append(
            [name, f"{s['dates'][0]} .. {s['dates'][1]}", num(s["rows"]), pct(s["prevalence"])]
        )
    return md_table(
        ["split", "dates", f"{_entity(contract)} rows", f"{_label(contract)} rate"], rows
    )


def render_evaluation_md(
    metrics: dict,
    meta: dict,
    contract: dict,
    narrative: EvaluationNarrative,
    source: str,
    grounding: dict,
) -> str:
    cands = metrics["candidates"]
    chosen = metrics["selected_model"]
    label, entity = _label(contract), _entity(contract)
    cv_rows = [
        [
            name,
            _pm(c["cv"]["roc_auc_mean"], c["cv"]["roc_auc_std"]),
            _pm(c["cv"]["pr_auc_mean"], c["cv"]["pr_auc_std"]),
        ]
        for name, c in cands.items()
    ]
    folds = cands[chosen]["cv"]["folds"]
    fold_rows = [
        [
            f["fold"],
            num(f["train_rows"]),
            num(f["val_rows"]),
            pct(f["val_prevalence"]),
            _f3(f["roc_auc"]),
            _f3(f["pr_auc"]),
        ]
        for f in folds
    ]
    test_rows = [
        [
            name,
            _f3(c["test"]["roc_auc"]),
            _f3(c["test"]["pr_auc"]),
            pct(c["test"]["precision_at_top10pct"]),
            "n/a"
            if c["test"]["lift_at_top10pct"] is None
            else f"{c['test']['lift_at_top10pct']:.2f}x",
            _f3(c["test"]["brier"]),
        ]
        for name, c in cands.items()
    ]
    t = cands[chosen]["test"]
    op_rows = [
        ["threshold (tuned on out-of-fold F1)", f"{t['threshold']:.3f}"],
        [f"{entity} rows flagged", pct(t["flagged_share"])],
        ["precision", pct(t["precision"])],
        ["recall", pct(t["recall"])],
        ["F1", _f3(t["f1"])],
        ["precision in riskiest 10%", pct(t["precision_at_top10pct"])],
    ]
    importance = metrics["permutation_importance"]["features"][:10]
    imp_rows = [
        [i + 1, f["feature"], f"{f['importance_mean']:.4f}"] for i, f in enumerate(importance)
    ]
    sel = metrics["selection"]
    excluded = meta.get("excluded_splits") or {}
    parts = [
        f"# Evaluation report: {label} risk per {entity}",
        "",
        "## Setup (computed)",
        "",
        f"- **Task:** binary classification of `{contract['target']['name']}` "
        f"({contract['target']['definition']}).",
        f"- **Validation:** {metrics['cv_scheme']}; threshold tuned on out-of-fold predictions; "
        "the test window is scored once.",
        f"- **Excluded windows:** {num(excluded.get('rows', 0))} rows outside train and test "
        f"({pct(excluded.get('positive_rate'))} {label}) are not used in the metrics.",
        "",
        split_table(meta, contract),
        "",
        "## Cross-validation on the train window (computed)",
        "",
        md_table(["model", "ROC-AUC (mean ± std)", "PR-AUC (mean ± std)"], cv_rows),
        "",
        f"Folds for the selected model ({chosen}); note the shifting {label} rate:",
        "",
        md_table(
            ["fold", "train rows", "validation rows", f"{label} rate", "ROC-AUC", "PR-AUC"],
            fold_rows,
        ),
        "",
        "## Test window results (computed)",
        "",
        md_table(
            ["model", "ROC-AUC", "PR-AUC", "precision top 10%", "lift top 10%", "Brier"], test_rows
        ),
        "",
        "## Model selection",
        "",
        f"- **Rule:** {sel['rule']}.",
        "- **CV PR-AUC:** "
        + ", ".join(f"{k} {v:.4f}" for k, v in sorted(sel["scores_cv_pr_auc"].items())),
        f"- **Selected:** `{chosen}` (margin {sel['selection_margin']:.4f} PR-AUC over the "
        "runner-up).",
        "",
        "## Operating point of the selected model (test window)",
        "",
        md_table(["measure", "value"], op_rows),
        "",
        "## Most important features (permutation importance, test window)",
        "",
        md_table(["rank", "feature", "drop in PR-AUC"], imp_rows),
        "",
        "## Figures",
        "",
        "\n".join(f"- `figures/{name}`" for name in metrics["figures"]),
        "",
        "## Limitations of this evaluation",
        "",
        bullets(
            [
                f"Prevalence shift: {pct(meta['split']['train']['prevalence'])} {label} in train "
                f"vs {pct(meta['split']['test']['prevalence'])} in test, so absolute precision "
                "and the threshold do not transfer automatically.",
                "One test window; results may differ in other periods.",
            ]
        ),
        "",
        "## Reviewer commentary",
        "",
        f"**Summary.** {narrative.executive_summary}",
        "",
        f"**Comparison.** {narrative.comparison_summary}",
        "",
        "**Strengths**",
        "",
        bullets(narrative.strengths),
        "",
        "**Weaknesses**",
        "",
        bullets(narrative.weaknesses),
        "",
        f"**Threshold recommendation.** {narrative.threshold_recommendation}",
        "",
        "**Next steps**",
        "",
        bullets(narrative.next_steps),
    ]
    return "\n".join(parts) + "\n" + footer(source, grounding)


def render_model_card_md(
    metrics: dict,
    meta: dict,
    contract: dict,
    narrative: ModelCardNarrative,
    template: ModelCardNarrative,
    source: str,
    grounding: dict,
) -> str:
    chosen = metrics["selected_model"]
    c = metrics["candidates"][chosen]
    t = c["test"]
    base = metrics["candidates"]["baseline_prior"]["test"]
    versions = meta["versions"]
    label, entity = _label(contract), _entity(contract)
    stage = contract["feature_rules"]["feature_available_at"]
    facts = md_table(
        ["fact", "value"],
        [
            [
                "training rows",
                f"{num(meta['split']['train']['rows'])} "
                f"({meta['split']['train']['dates'][0]} .. {meta['split']['train']['dates'][1]})",
            ],
            [
                "test rows",
                f"{num(meta['split']['test']['rows'])} "
                f"({meta['split']['test']['dates'][0]} .. {meta['split']['test']['dates'][1]})",
            ],
            [
                f"{label} rate train / test",
                f"{pct(meta['split']['train']['prevalence'])} / "
                f"{pct(meta['split']['test']['prevalence'])}",
            ],
            ["features", f"{len(meta['model_features'])} {stage} features"],
            [
                "dataset contract",
                f"v{meta['contract_version']} sha256 {meta['contract_sha256'][:12]}",
            ],
            ["raw data", str(contract.get("provenance", {}).get("source") or "see provenance")],
            ["privacy", str(contract.get("privacy", {}).get("note") or "see the contract")],
        ],
    )
    metric_rows = [
        ["CV PR-AUC (mean ± std)", _pm(c["cv"]["pr_auc_mean"], c["cv"]["pr_auc_std"]), "n/a"],
        ["test ROC-AUC", _f3(t["roc_auc"]), _f3(base["roc_auc"])],
        ["test PR-AUC", _f3(t["pr_auc"]), _f3(base["pr_auc"])],
        [
            "precision in riskiest 10%",
            pct(t["precision_at_top10pct"]),
            pct(base["precision_at_top10pct"]),
        ],
        ["Brier score", _f3(t["brier"]), _f3(base["brier"])],
        [
            f"precision / recall at threshold {t['threshold']:.3f}",
            f"{pct(t['precision'])} / {pct(t['recall'])}",
            "n/a",
        ],
    ]
    parts = [
        f"# Model card: {label} risk per {entity}",
        "",
        "## Model details",
        "",
        f"- **Model:** `{chosen}` ({meta['estimator']}): {meta['description']}.",
        f"- **Selected by:** {metrics['selection']['rule']}.",
        f"- **Decision threshold:** {meta['threshold']:.3f} (tuned on out-of-fold F1).",
        f"- **Reproducibility:** seed {meta['seed']}, {meta['thread_limit']} threads, scikit-learn "
        f"{versions['scikit_learn']}, Python {versions['python']}.",
        f"- **Contract:** dataset_contract v{meta['contract_version']}, sha256 "
        f"`{meta['contract_sha256']}`.",
        "",
        "## Model purpose",
        "",
        narrative.purpose,
        "",
        f"- **Intended users:** {narrative.intended_users}",
        "- **Out of scope:**",
        "\n".join(
            f"  - {x}"
            for x in merge_items(
                template.out_of_scope, narrative.out_of_scope if narrative is not template else None
            )
        ),
        "",
        "## Training data summary",
        "",
        facts,
        "",
        narrative.training_data_summary,
        "",
        "## Metrics",
        "",
        md_table(["metric", chosen, "baseline_prior"], metric_rows),
        "",
        "## Limitations",
        "",
        bullets(
            merge_items(
                template.limitations, narrative.limitations if narrative is not template else None
            )
        ),
        "",
        "## Ethical considerations",
        "",
        bullets(
            merge_items(
                template.ethical_considerations,
                narrative.ethical_considerations if narrative is not template else None,
            )
        ),
        "",
        "## Monitoring",
        "",
        bullets(
            merge_items(
                template.monitoring_recommendations,
                narrative.monitoring_recommendations if narrative is not template else None,
            )
        ),
        "",
        "## How to use",
        "",
        "Run the API (`uv run pulse-api`) and POST records whose fields are the model features "
        f"listed in feature_spec.json ({len(meta['model_features'])} fields); the response gives "
        "a probability, a risk level and a recommended action per record.",
    ]
    return "\n".join(parts) + "\n" + footer(source, grounding)
