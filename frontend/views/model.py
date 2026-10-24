"""Model: how good the selected model is, and why it was chosen (Data Scientist crew outputs)."""

import pandas as pd
import streamlit as st

from components import cards, charts, layout
from components.format import num, pct, times
from services import artifacts as art

layout.page_header(
    "How good is the model?",
    "Three candidates were trained on the train window and tested on a later window the model "
    "never saw.",
    kicker="Model · Data Scientist crew",
)
metrics, meta = art.metrics(), art.model_meta()
if not metrics or not meta:
    layout.empty_state("No model in artifacts/ yet.")
    st.stop()

chosen = metrics["selected_model"]
candidates = metrics["candidates"]
test = candidates[chosen]["test"]
threshold = float(meta["threshold"])
label = charts.model_label(chosen)

KPI_HEIGHT = 132
c1, c2, c3, c4 = st.columns(4)
cards.kpi(
    c1,
    "Test ROC-AUC",
    f"{test['roc_auc']:.3f}",
    delta=f"{test['roc_auc'] - 0.5:+.3f} vs chance",
    help="Probability that a random positive row scores above a random negative row.",
    height=KPI_HEIGHT,
)
cards.kpi(
    c2,
    "Test PR-AUC",
    f"{test['pr_auc']:.3f}",
    delta=f"{times(test['pr_auc'] / test['prevalence'])} the base rate",
    help=f"Average precision; a model with no skill scores the base rate, {pct(test['prevalence'])}.",
    height=KPI_HEIGHT,
)
cards.kpi(
    c3,
    "Positives in the riskiest 10%",
    pct(test["precision_at_top10pct"]),
    delta=f"{times(test['lift_at_top10pct'])} the base rate",
    help="Share of positives among the 10% of test rows with the highest scores.",
    height=KPI_HEIGHT,
)
cards.kpi(
    c4,
    "Positives caught",
    pct(test["recall"]),
    delta=f"{pct(test['flagged_share'])} of rows flagged",
    delta_color="off",
    help=f"At the decision threshold {threshold:.3f} the model flags "
    f"{pct(test['flagged_share'])} of rows; {pct(test['precision'])} of flagged rows are positive.",
    height=KPI_HEIGHT,
)

# Comparison -----------------------------------------------------------------------------------
selection = metrics["selection"]
layout.section("Model comparison", f"Selection rule: {selection['rule']}.")
left, right = st.columns(2)
with left:
    layout.chart_card(
        "Positives among the riskiest 10%",
        charts.precision_at_top(candidates, chosen, test["prevalence"]),
        f"{label} was selected with a cross-validated PR-AUC of "
        f"{selection['scores_cv_pr_auc'][chosen]:.3f} (margin {selection['selection_margin']:.3f}).",
    )
with right, st.container(border=True, height="stretch"):
    st.markdown("**Evaluation design: a time split**")
    split = metrics["split"]
    order = [k for k in ("train", "test") if k in split] + sorted(
        k for k in split if k not in ("train", "test")
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Split": name.replace("_", " "),
                    "Dates": " to ".join(split[name]["dates"]),
                    "Rows": num(split[name]["rows"]),
                    "Positive rate": pct(split[name]["prevalence"]),
                }
                for name in order
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        ":material/lightbulb: Cross-validation uses three time-ordered folds inside the train "
        "window; windows other than train and test are excluded from the metrics."
    )

with st.container(border=True):
    st.markdown("**All candidates, test window**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Model": charts.model_label(name) + ("  ★ selected" if name == chosen else ""),
                    "CV PR-AUC": f"{c['cv']['pr_auc_mean']:.3f} ± {c['cv']['pr_auc_std']:.3f}",
                    "ROC-AUC": f"{c['test']['roc_auc']:.3f}",
                    "PR-AUC": f"{c['test']['pr_auc']:.3f}",
                    "Positives, riskiest 10%": pct(c["test"]["precision_at_top10pct"]),
                    "Lift": f"{c['test']['lift_at_top10pct']:.2f}x",
                    "Brier (lower is better)": f"{c['test']['brier']:.4f}",
                }
                for name, c in candidates.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )

# Curves -----------------------------------------------------------------------------------------
curves = art.model_curves(tuple(candidates), chosen)
if curves is not None:
    roc, pr, calibration, prevalence = curves
    for frame in (roc, pr):
        frame["model"] = frame["model"].map(charts.model_label)
    layout.section("Ranking quality", "Test window; hover the curves for exact values")
    left, right = st.columns(2)
    with left:
        layout.chart_card(
            "ROC curves",
            charts.roc_curves(roc),
            f"{label}: ROC-AUC {test['roc_auc']:.3f}; the dashed diagonal is chance.",
        )
    best_pr = max(candidates, key=lambda n: candidates[n]["test"]["pr_auc"])
    with right:
        layout.chart_card(
            "Precision-recall curves",
            charts.pr_curves(pr, prevalence),
            f"The dashed line is the {pct(prevalence)} base rate; "
            f"{charts.model_label(best_pr).lower()} has the highest test PR-AUC "
            f"({candidates[best_pr]['test']['pr_auc']:.3f}).",
        )
    layout.section("Calibration and drivers")
    left, right = st.columns(2)
    with left:
        if not calibration.empty:
            over = calibration["mean_predicted"].mean() > calibration["observed"].mean()
            train_rate = metrics["split"]["train"]["prevalence"]
            test_rate = metrics["split"]["test"]["prevalence"]
            layout.chart_card(
                f"Calibration of the {label.lower()} model",
                charts.calibration(calibration),
                f"Trained when {pct(train_rate)} of rows were positive and tested at "
                f"{pct(test_rate)}, the model {'over' if over else 'under'}-predicts (points "
                f"{'below' if over else 'above'} the diagonal). Rankings still hold, and the "
                "threshold is re-tuned on recent data.",
            )
    features = metrics["permutation_importance"]["features"]
    with right:
        layout.chart_card(
            "What the model relies on",
            charts.importance(features),
            "Permutation importance on the test window. Strongest signals: "
            + ", ".join(f["feature"] for f in features[:4])
            + ".",
        )

# Threshold --------------------------------------------------------------------------------------
layout.section(
    "Decision threshold",
    "Tuned on out-of-fold training predictions (best F1), never on the test window.",
)
c1, c2, c3, c4 = st.columns(4)
cards.kpi(c1, "Threshold", f"{threshold:.3f}")
cards.kpi(c2, "Rows flagged", pct(test["flagged_share"]))
cards.kpi(c3, "Precision when flagged", pct(test["precision"]))
cards.kpi(c4, "Recall", pct(test["recall"]))

# Documents --------------------------------------------------------------------------------------
layout.section("Documents", "Written by the pipeline; narratives by the Data Scientist crew")
tab_card, tab_eval, tab_spec = st.tabs(["Model card", "Evaluation report", "Feature spec"])


def _without_title(markdown: str | None, missing: str) -> str:
    if not markdown:
        return f"_{missing} not found_"
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines)


with tab_card:
    st.markdown(_without_title(art.load_text("model_card.md"), "model_card.md"))
with tab_eval:
    st.markdown(_without_title(art.load_text("evaluation_report.md"), "evaluation_report.md"))
with tab_spec:
    spec = art.load_json("feature_spec.json")
    if spec:
        st.json(spec, expanded=1)
layout.footer()
