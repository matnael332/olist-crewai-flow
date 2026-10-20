import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401
from services.artifacts import figure, load_json, load_text, pct

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
                "CV PR-AUC": f"{c['cv']['pr_auc_mean']:.3f}",
                "CV ROC-AUC": f"{c['cv']['roc_auc_mean']:.3f}",
                "test ROC-AUC": f"{c['test']['roc_auc']:.3f}",
                "test PR-AUC": f"{c['test']['pr_auc']:.3f}",
                "precision, riskiest 10%": pct(c["test"]["precision_at_top10pct"]),
                "lift, riskiest 10%": f"{c['test']['lift_at_top10pct']:.2f}x",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    # Chronological order (the JSON keys are sorted alphabetically).
    order = [k for k in ("train", "test", "censored_buffer") if k in metrics["split"]]
    order += [k for k in metrics["split"] if k not in order]
    split = pd.DataFrame(
        [
            {
                "split": k,
                "dates": " to ".join(metrics["split"][k]["dates"]),
                "orders": f"{metrics['split'][k]['rows']:,}",
                "late rate": pct(metrics["split"][k]["prevalence"]),
            }
            for k in order
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
