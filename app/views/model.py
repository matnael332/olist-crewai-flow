import _bootstrap  # noqa: F401
import pandas as pd
import streamlit as st
from data_access import figure, load_json, load_text, pct

st.title("Model: late-delivery risk at checkout")
metrics = load_json("metrics.json")
meta = load_json("model_meta.json")
if not metrics or not meta:
    st.info("No model in artifacts/ yet.")
    st.stop()

chosen = metrics["selected_model"]
st.markdown(
    f"Selected model: **{chosen}** ({meta['estimator']}), decision threshold "
    f"{meta['threshold']:.3f}. {metrics['selection']['rule'].capitalize()}."
)
tab_metrics, tab_eval, tab_card, tab_spec = st.tabs(
    ["Metrics", "evaluation_report.md", "model_card.md", "feature_spec.json"]
)
with tab_metrics:
    rows = []
    for name, c in metrics["candidates"].items():
        rows.append(
            {
                "model": name + (" (selected)" if name == chosen else ""),
                "CV PR-AUC": c["cv"]["pr_auc_mean"],
                "CV ROC-AUC": c["cv"]["roc_auc_mean"],
                "test ROC-AUC": c["test"]["roc_auc"],
                "test PR-AUC": c["test"]["pr_auc"],
                "precision, riskiest 10%": pct(c["test"]["precision_at_top10pct"]),
                "lift, riskiest 10%": c["test"]["lift_at_top10pct"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    split = pd.DataFrame(
        [
            {
                "split": k,
                "dates": " .. ".join(v["dates"]),
                "orders": v["rows"],
                "late rate": pct(v["prevalence"]),
            }
            for k, v in metrics["split"].items()
        ]
    )
    st.caption(
        "The late rate drifts between periods, so thresholds must be re-tuned on recent data."
    )
    st.dataframe(split, hide_index=True, width="stretch")
    cols = st.columns(2)
    for i, name in enumerate(
        [
            "roc_curves.png",
            "pr_curves.png",
            "calibration_selected.png",
            "feature_importance_selected.png",
        ]
    ):
        path = figure(name)
        if path:
            cols[i % 2].image(str(path), width="stretch")
with tab_eval:
    st.markdown(load_text("evaluation_report.md") or "No evaluation report.")
with tab_card:
    st.markdown(load_text("model_card.md") or "No model card.")
with tab_spec:
    st.json(load_json("feature_spec.json") or {}, expanded=False)
