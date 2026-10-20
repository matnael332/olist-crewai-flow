"""Order scoring for the Risk scoring page.

Scores in-process with the backend's ``olist_api.model.ModelBundle`` (the same code the Flask API
serves, so there is nothing else to host), or over HTTP when ``API_URL`` is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import requests
import streamlit as st

import _bootstrap  # noqa: F401  (puts backend/ on sys.path before the olist_api import)
from components.format import pct
from olist_api.model import ModelBundle
from services.artifacts import ARTIFACTS, ROOT

API_URL = os.getenv("API_URL", "").rstrip("/")

_BASE_ORDER = {
    "purchase_date": date(2018, 6, 12),
    "purchase_time": time(14, 0),
    "promise_date": date(2018, 7, 5),
    "seller_state": "SP",
    "primary_category": "furniture_decor",
    "n_items": 1,
    "total_price": 129.9,
    "total_freight": 21.5,
    "total_weight_g": 1200.0,
    "payment_type": "boleto",
    "max_installments": 1,
}
PRESETS: dict[str, dict] = {
    "Long distance: SP to BA": {**_BASE_ORDER, "customer_state": "BA"},
    "Same state: SP to SP": {**_BASE_ORDER, "customer_state": "SP"},
}
CUSTOM = "Custom order"


@st.cache_resource(show_spinner="Loading the model...")
def _bundle(mtime: float) -> ModelBundle:
    return ModelBundle(ARTIFACTS, ROOT / "data" / "lookups")


def bundle() -> ModelBundle:
    path = ARTIFACTS / "model.joblib"
    return _bundle(path.stat().st_mtime if path.exists() else 0.0)


def to_record(order: dict) -> dict:
    """Form values -> a /predict record (the API's input schema)."""
    purchase = datetime.combine(order["purchase_date"], order["purchase_time"])
    return {
        "n_items": int(order["n_items"]),
        "total_price": float(order["total_price"]),
        "total_freight": float(order["total_freight"]),
        "total_weight_g": float(order["total_weight_g"]),
        "customer_state": order["customer_state"],
        "seller_state": order["seller_state"],
        "primary_category": order["primary_category"],
        "payment_type": order["payment_type"],
        "max_installments": int(order["max_installments"]),
        "purchase_ts": purchase.strftime("%Y-%m-%d %H:%M:%S"),
        "estimated_delivery_date": order["promise_date"].strftime("%Y-%m-%d"),
    }


def score(records: list[dict]) -> tuple[list[dict], list]:
    """(predictions, errors) for one or more records."""
    if API_URL:
        response = requests.post(f"{API_URL}/predict", json={"records": records}, timeout=15)
        body = response.json()
        if response.status_code != 200:
            return [], body.get("details") or [body]
        return body["predictions"], []
    model = bundle()
    if not model.ok:
        return [], [{"message": f"model unavailable: {model.error}"}]
    predictions, errors, _ = model.predict(records)
    return predictions, errors


@dataclass(frozen=True)
class RiskLevel:
    label: str
    color: str  # badge color
    bar: str  # chart color
    message: str
    kind: str  # "warning" | "info" | "success": the Streamlit alert used for the message


# How each backend level looks; the level itself and its action come from olist_api.risk.
_LOOK = {
    "high": ("High risk", "red", "#DC2626", "warning"),
    "elevated": ("Elevated risk", "orange", "#F59E0B", "info"),
    "normal": ("Normal risk", "green", "#16A34A", "success"),
}


def risk_level(prediction: dict, threshold: float) -> RiskLevel:
    """Presentation of the backend's decision (prediction["risk_level"], recommended_action)."""
    level = prediction.get("risk_level")
    if level not in _LOOK:  # an older API without risk_level: fall back to its flags
        level = (
            "high"
            if prediction["top_10pct_risk"]
            else "elevated"
            if prediction["predicted_late"]
            else "normal"
        )
    why = {
        "high": "Among the riskiest 10% of orders.",
        "elevated": f"Above the model's decision threshold ({pct(threshold)}), but outside the "
        "riskiest 10% of orders.",
        "normal": f"Below the model's decision threshold ({pct(threshold)}).",
    }[level]
    action = prediction.get("recommended_action") or ""
    label, color, bar, kind = _LOOK[level]
    return RiskLevel(label, color, bar, f"{why} {action}".strip(), kind)


def what_if(record: dict, base_probability: float) -> list[dict]:
    """Re-score the order with one change at a time: what would lower (or raise) the risk?"""
    scenarios = []
    if record["seller_state"] != record["customer_state"]:
        scenarios.append(
            ("Seller in the customer's state", {**record, "seller_state": record["customer_state"]})
        )
    later = date.fromisoformat(record["estimated_delivery_date"]) + timedelta(days=7)
    scenarios.append(
        ("Promise 7 more days", {**record, "estimated_delivery_date": later.isoformat()})
    )
    if record["payment_type"] != "credit_card":
        scenarios.append(("Pay by credit card", {**record, "payment_type": "credit_card"}))
    predictions, errors = score([changed for _, changed in scenarios])
    if errors:
        return []
    return [
        {
            "scenario": name,
            "probability": p["late_probability"],
            "delta": p["late_probability"] - base_probability,
        }
        for (name, _), p in zip(scenarios, predictions, strict=True)
    ]
