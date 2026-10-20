"""Risk scoring: score an order at checkout and see what would change the risk."""

import streamlit as st

from components import charts, layout
from components.format import pct, points, times
from services import artifacts as art
from services import scoring

layout.page_header(
    "Score an order at checkout",
    "Pick a preset or enter an order. The model only uses what is known when the customer pays: "
    "basket, package, payment, the two states and the promised delivery date.",
    kicker="Risk scoring",
)
contract = art.contract()
model = scoring.bundle()
if not contract or not model.ok:
    layout.empty_state(f"The model is not available ({model.error or 'no artifacts'}).")
    st.stop()

allowed = {c["name"]: c.get("allowed_values") or [] for c in contract["columns"]}
threshold = float(model.meta["threshold"])
first_preset = next(iter(scoring.PRESETS))
choice = st.segmented_control(
    "Start from",
    [*scoring.PRESETS, scoring.CUSTOM],
    default=first_preset,
    key="preset",
)
choice = choice or first_preset
defaults = scoring.PRESETS.get(choice, scoring.PRESETS[first_preset])
key = choice.split(":")[0].lower().replace(" ", "-")  # fresh widgets (and defaults) per preset


def _index(options: list, value) -> int:
    return options.index(value) if value in options else 0


form_column, result_column = st.columns([3, 2], gap="large")
with form_column, st.form(f"order-{key}"):
    st.markdown("**Order**")
    c1, c2, c3 = st.columns(3)
    purchase_date = c1.date_input("Purchase date", defaults["purchase_date"], key=f"pd-{key}")
    purchase_time = c2.time_input("Purchase time", defaults["purchase_time"], key=f"pt-{key}")
    promise_date = c3.date_input("Promised delivery", defaults["promise_date"], key=f"pr-{key}")
    st.markdown("**Shipping**")
    c1, c2, c3 = st.columns(3)
    states, sellers = allowed["customer_state"], allowed["seller_state"]
    customer_state = c1.selectbox(
        "Customer state", states, _index(states, defaults["customer_state"]), key=f"cs-{key}"
    )
    seller_state = c2.selectbox(
        "Seller state", sellers, _index(sellers, defaults["seller_state"]), key=f"ss-{key}"
    )
    weight = c3.number_input(
        "Total weight (g)", 0.0, 50000.0, defaults["total_weight_g"], step=100.0, key=f"w-{key}"
    )
    st.markdown("**Basket and payment**")
    c1, c2, c3 = st.columns(3)
    categories = allowed["primary_category"]
    category = c1.selectbox(
        "Category", categories, _index(categories, defaults["primary_category"]), key=f"c-{key}"
    )
    n_items = c2.number_input("Items", 1, 30, defaults["n_items"], key=f"n-{key}")
    price = c3.number_input(
        "Price (BRL)", 0.0, 15000.0, defaults["total_price"], step=10.0, key=f"p-{key}"
    )
    c1, c2, c3 = st.columns(3)
    freight = c1.number_input(
        "Freight (BRL)", 0.0, 500.0, defaults["total_freight"], step=1.0, key=f"f-{key}"
    )
    payments = allowed["payment_type"]
    payment = c2.selectbox(
        "Payment type", payments, _index(payments, defaults["payment_type"]), key=f"pay-{key}"
    )
    installments = c3.number_input(
        "Installments", 0, 24, defaults["max_installments"], key=f"i-{key}"
    )
    st.form_submit_button("Score order", type="primary", icon=":material/speed:", width="stretch")

with result_column:
    if promise_date < purchase_date:
        st.error("The promised delivery date is before the purchase date.")
        st.stop()
    record = scoring.to_record(
        {
            "purchase_date": purchase_date,
            "purchase_time": purchase_time,
            "promise_date": promise_date,
            "customer_state": customer_state,
            "seller_state": seller_state,
            "primary_category": category,
            "n_items": n_items,
            "total_price": price,
            "total_freight": freight,
            "total_weight_g": weight,
            "payment_type": payment,
            "max_installments": installments,
        }
    )
    predictions, errors = scoring.score([record])
    if errors:
        st.error("The order did not pass the dataset contract checks:")
        st.json(errors)
        st.stop()
    prediction = predictions[0]
    probability = prediction["late_probability"]
    level = scoring.risk_level(prediction, threshold)
    with st.container(border=True):
        st.metric(
            "Late-delivery probability",
            pct(probability),
            delta=f"{points(probability - model.test_prevalence)} vs the base rate "
            f"({times(probability / model.test_prevalence)})",
            delta_color="inverse",
            help=f"Test-window base rate {pct(model.test_prevalence)}; decision threshold "
            f"{pct(threshold)}; the riskiest 10% of orders score above {pct(model.top10_cutoff)}.",
        )
        st.altair_chart(
            charts.risk_bullet(
                probability, model.test_prevalence, threshold, model.top10_cutoff, level.bar
            ),
            width="stretch",
        )
        getattr(st, level.kind)(f"**{level.label}.** {level.message}")

    st.markdown("**What would change the risk?**")
    st.caption("The same order re-scored with one change at a time.")
    for scenario in scoring.what_if(record, probability):
        st.metric(
            scenario["scenario"],
            pct(scenario["probability"]),
            delta=points(scenario["delta"]),
            delta_color="inverse",
            border=True,
        )
    st.caption(
        ":material/shield: Scores rank orders for proactive help, never to deny service or "
        "penalize customers or sellers."
        + (f" Scored through the API at {scoring.API_URL}." if scoring.API_URL else "")
    )
layout.footer()
