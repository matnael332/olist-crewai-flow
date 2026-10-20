"""Overview: the business question, the answer, and how the system got there."""

import streamlit as st

from components import cards, layout, pipeline
from components.format import num, pct, times
from services import artifacts as art

layout.page_header(
    "Will this order arrive late?",
    "Late deliveries drive most 1-2 star reviews on the Olist marketplace. This product scores "
    "every order at checkout, so operations can act before the customer is let down.",
    kicker="Olist late-delivery AI workflow",
)

eda, metrics = art.eda(), art.metrics()
if not eda or not metrics:
    layout.empty_state("No pipeline artifacts yet. Run `uv run kickoff` to produce them.")
    st.stop()

trusted, run_id = art.artifacts_valid()
gates = art.gate_summary(art.validation())
run_status = art.status() or {}
promoted = trusted and run_status.get("status") == "success"
if trusted:
    all_passed = len(gates) == 3 and all(g["passed"] for g in gates.values())
    with st.container(horizontal=True, gap="small"):
        cards.badge(
            "All 3 quality gates passed" if all_passed else "A quality gate failed", ok=all_passed
        )
        cards.badge(f"Run {run_id}", icon=":material/history:")
        narrated = run_status.get("mode") == "llm"
        cards.badge(
            "Narratives by Gemini agents" if narrated else "Template narratives",
            icon=":material/smart_toy:",
        )
else:
    st.warning(
        f"These numbers are not from a promoted pipeline run ({run_id}).", icon=":material/warning:"
    )

dp, ao = eda["delivered_population"], eda["all_orders"]
chosen = metrics["selected_model"]
test = metrics["candidates"][chosen]["test"]
reviews = dp["reviews"]

st.space("small")
KPI_HEIGHT = 132
c1, c2, c3, c4 = st.columns(4)
cards.kpi(
    c1,
    "Orders delivered late",
    pct(dp["late_rate"]),
    help=f"{num(dp['late_orders'])} of {num(dp['n_orders'])} delivered orders, purchases "
    "Jan 2017 to Aug 2018.",
    height=KPI_HEIGHT,
)
cards.kpi(
    c2,
    "Bad reviews: late / on time",
    f"{pct(reviews['bad_review_rate_late'], 0)} / {pct(reviews['bad_review_rate_on_time'], 0)}",
    help=f"{pct(reviews['bad_review_rate_late'])} of late orders get a 1-2 star review, against "
    f"{pct(reviews['bad_review_rate_on_time'])} of orders that arrive on time.",
    height=KPI_HEIGHT,
)
cards.kpi(
    c3,
    "Late rate in the riskiest 10%",
    pct(test["precision_at_top10pct"]),
    help=f"Test window (May-Jul 2018): the 10% of orders the model scores highest are late "
    f"{times(test['lift_at_top10pct'])} as often as the {pct(test['prevalence'])} average.",
    delta=f"{times(test['lift_at_top10pct'])} the base rate",
    height=KPI_HEIGHT,
)
cards.kpi(
    c4,
    "Test ROC-AUC",
    f"{test['roc_auc']:.3f}",
    help="Test window May-Jul 2018. Probability that a random late order scores higher than a "
    "random on-time order (0.5 = chance).",
    delta=f"{test['roc_auc'] - 0.5:+.3f} vs chance",
    height=KPI_HEIGHT,
)

layout.section(
    "How it works",
    "A CrewAI Flow runs two AI agent crews. Tested Python computes every number; agents inspect "
    "and explain; three quality gates decide whether a run is published.",
)
pipeline.stepper(gates, promoted)

layout.section("What the data says")
peak_day, second_day = ao["peak_day"], ao["second_busiest_day"]
states = sorted(
    (s for s in dp["late_rate_by_state"] if s["orders"] >= 0.004 * dp["n_orders"]),
    key=lambda s: s["late_rate"],
)
worst, best = states[-1], states[0]
delivery = dp["delivery"]
c1, c2, c3 = st.columns(3)
cards.insight(
    c1,
    "Black Friday spike",
    f"{num(peak_day['orders'])} orders in one day",
    f"on {peak_day['date']}, {times(peak_day['orders'] / second_day['orders'])} the next busiest "
    "day. Peaks strain the carrier network.",
    ":material/local_fire_department:",
)
cards.insight(
    c2,
    "Geography drives delays",
    f"{worst['state']} {pct(worst['late_rate'])} vs {best['state']} {pct(best['late_rate'])}",
    "late rate in the worst and best states. North and Northeast customers wait longest.",
    ":material/map:",
)
cards.insight(
    c3,
    "Promises are padded",
    f"{delivery['median_days']:.0f} vs {delivery['median_promised_days']:.0f} days",
    f"median actual vs promised delivery time, yet {pct(dp['late_rate'])} of orders still arrive "
    "late.",
    ":material/schedule:",
)

layout.section("Explore")
c1, c2, c3 = st.columns(3)
c1.page_link("views/scoring.py", label="Score an order at checkout", icon=":material/speed:")
c2.page_link("views/insights.py", label="See what happened", icon=":material/insights:")
c3.page_link("views/model.py", label="How good is the model?", icon=":material/model_training:")
layout.footer()
