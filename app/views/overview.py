import _bootstrap  # noqa: F401
import pandas as pd
import streamlit as st
from data_access import artifacts_valid, load_json, load_text, pct

st.title("Will this order arrive late?")
st.markdown(
    "An industry-style AI workflow on the **Olist Brazilian e-commerce** dataset: a CrewAI Flow "
    "hands off from a **Data Analyst crew** (cleaning, EDA, insights, dataset contract) to a "
    "**Data Scientist crew** (features, model comparison, evaluation, model card), with "
    "deterministic validation gates between them."
)

valid, detail = artifacts_valid()
status = load_json("run_status.json") or {}
if not valid:
    st.warning(f"No trustworthy promoted run: {detail}")
elif status.get("status") == "success":
    sources = sorted(set((status.get("narrative_sources") or {}).values())) or ["template"]
    st.success(
        f"Promoted run `{status['run_id']}` succeeded ({status.get('finished_at', '')}); "
        f"mode **{status.get('mode')}**, narratives: {', '.join(sources)}."
    )
else:
    st.error(f"Run `{status.get('run_id')}` {status.get('status')} at {status.get('stage')}")

summary = load_json("eda_summary.json") or {}
metrics = load_json("metrics.json") or {}
ao = summary.get("all_orders", {})
dp = summary.get("delivered_population", {})
chosen = metrics.get("selected_model")
test = metrics.get("candidates", {}).get(chosen, {}).get("test", {})

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Orders (all statuses)", f"{ao.get('n_orders', 0):,}")
c2.metric("Delivered orders", f"{dp.get('n_orders', 0):,}")
c3.metric("Late rate (delivered)", pct(dp.get("late_rate")))
c4.metric("Median delivery", f"{dp.get('delivery', {}).get('median_days', 0):.0f} days")
c5.metric("Model ROC-AUC (test)", f"{test.get('roc_auc', 0):.3f}")
c6.metric(
    "Precision, riskiest 10%",
    pct(test.get("precision_at_top10pct")),
    help=f"vs {pct(test.get('prevalence'))} test late rate",
)

left, right = st.columns([3, 2])
with left:
    st.subheader("The Flow")
    gates = (load_json("validation_report.json") or {}).get("gates", {})

    def color(gate: str) -> str:
        if gate not in gates:
            return "lightgrey"
        return "palegreen" if gates[gate]["passed"] else "lightcoral"

    st.graphviz_chart(
        f"""
        digraph flow {{
          rankdir=LR; node [shape=box, style="rounded,filled", fillcolor=white, fontsize=10];
          raw [label="9 raw tables\\n(pinned + hashed)"];
          crew1 [label="Crew 1: Data Analyst\\n6 agents", fillcolor="#dbeafe"];
          g1 [label="gate_contract", shape=diamond, fillcolor="{color("gate_contract")}"];
          crew2a [label="Crew 2 phase 1\\nfeatures", fillcolor="#fde68a"];
          g2 [label="gate_required\\n_features", shape=diamond, fillcolor="{color("gate_required_features")}"];
          crew2b [label="Crew 2 phase 2\\nmodels + card", fillcolor="#fde68a"];
          g3 [label="gate_artifacts", shape=diamond, fillcolor="{color("gate_artifacts")}"];
          out [label="artifacts/\\n(promoted)"];
          fail [label="fail_gracefully\\n(nothing promoted)", fillcolor="#fee2e2"];
          raw -> crew1 -> g1 -> crew2a -> g2 -> crew2b -> g3 -> out;
          g1 -> fail [style=dashed]; g2 -> fail [style=dashed]; g3 -> fail [style=dashed];
        }}
        """
    )
with right:
    st.subheader("Validation gates")
    if gates:
        st.dataframe(
            pd.DataFrame([{"gate": g, "result": r["summary"]} for g, r in gates.items()]),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No validation report yet.")

insights = load_text("insights.md")
if insights:
    st.subheader("What happened: insights")
    headline = insights.split("## Headline", 1)[-1].split("##", 1)[0].strip()
    st.info(headline)
    with st.expander("Full insights (insights.md)"):
        st.markdown(insights)
