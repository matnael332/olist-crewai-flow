import os
from datetime import date, datetime, time, timedelta

import _bootstrap  # noqa: F401
import requests
import streamlit as st
from data_access import ARTIFACTS, ROOT, load_json, pct

from api.model import ModelBundle

st.title("Score an order at checkout")
contract = load_json("dataset_contract.json")
if not contract:
    st.info("No model in artifacts/ yet.")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def bundle(mtime: float) -> ModelBundle:
    return ModelBundle(ARTIFACTS, ROOT / "data" / "lookups")


model = bundle((ARTIFACTS / "model.joblib").stat().st_mtime)
api_url = os.getenv("API_URL", "").rstrip("/")
st.caption(
    f"Scoring via the Flask API at `{api_url}`"
    if api_url
    else "Scoring locally with model.joblib (set API_URL to use the Flask API)."
)
allowed = {c["name"]: c.get("allowed_values") for c in contract["columns"]}

with st.form("order"):
    c1, c2, c3 = st.columns(3)
    purchase_day = c1.date_input("Purchase date", date(2018, 6, 12))
    purchase_time = c1.time_input("Purchase time", time(14, 0))
    promise = c1.date_input("Promised delivery date", date(2018, 7, 5))
    customer_state = c2.selectbox(
        "Customer state", allowed["customer_state"], index=allowed["customer_state"].index("BA")
    )
    seller_state = c2.selectbox(
        "Seller state", allowed["seller_state"], index=allowed["seller_state"].index("SP")
    )
    category = c2.selectbox(
        "Category (first item)",
        allowed["primary_category"],
        index=allowed["primary_category"].index("furniture_decor")
        if "furniture_decor" in allowed["primary_category"]
        else 0,
    )
    n_items = c3.number_input("Items", 1, 30, 1)
    price = c3.number_input("Price (BRL)", 0.0, 15000.0, 129.9)
    freight = c3.number_input("Freight (BRL)", 0.0, 500.0, 21.5)
    payment = c3.selectbox("Payment type", allowed["payment_type"])
    installments = c3.number_input("Installments", 0, 24, 1)
    weight = c1.number_input("Total weight (g)", 0.0, 50000.0, 1200.0)
    submitted = st.form_submit_button("Score order", type="primary")

if submitted:
    record = {
        "n_items": int(n_items),
        "total_price": float(price),
        "total_freight": float(freight),
        "total_weight_g": float(weight),
        "customer_state": customer_state,
        "seller_state": seller_state,
        "primary_category": category,
        "payment_type": payment,
        "max_installments": int(installments),
        "purchase_ts": datetime.combine(purchase_day, purchase_time).strftime("%Y-%m-%d %H:%M:%S"),
        "estimated_delivery_date": promise.strftime("%Y-%m-%d"),
    }
    if promise < purchase_day:
        st.error("The promised date is before the purchase date.")
        st.stop()
    if api_url:
        response = requests.post(f"{api_url}/predict", json={"records": [record]}, timeout=15)
        body = response.json()
        if response.status_code != 200:
            st.error(body)
            st.stop()
        prediction = body["predictions"][0]
    else:
        if not model.ok:
            st.error(f"Model unavailable: {model.error}")
            st.stop()
        predictions, errors, _ = model.predict([record])
        if errors:
            st.error(errors)
            st.stop()
        prediction = predictions[0]
    p = prediction["late_probability"]
    a, b, c = st.columns(3)
    a.metric(
        "Late-delivery probability",
        pct(p),
        help=f"test-window base rate {pct(model.test_prevalence)}",
    )
    b.metric("Riskiest 10% of orders?", "yes" if prediction["top_10pct_risk"] else "no")
    c.metric("Promise", f"{(promise - purchase_day).days} days")
    st.progress(min(1.0, p / max(model.top10_cutoff * 2, 0.01)))
    threshold = (getattr(model, "meta", None) or {}).get("threshold")
    at_threshold = f" ({pct(threshold)})" if threshold is not None else ""
    if prediction["top_10pct_risk"]:
        st.warning(
            "High risk: among the riskiest 10% of orders. Notify the customer proactively or "
            "upgrade the carrier."
        )
    elif prediction["predicted_late"]:
        st.info(
            f"Elevated risk: above the model's decision threshold{at_threshold}, but outside the "
            "riskiest 10% of orders. Keep an eye on this shipment."
        )
    else:
        st.success(f"Normal risk: below the model's decision threshold{at_threshold}.")
    st.caption(
        "Scores rank orders by risk; act on the riskiest 10% (precision there is about 3x the "
        "base rate). The model uses only information known at checkout."
    )
_ = timedelta  # keeps the import list explicit for date arithmetic above
