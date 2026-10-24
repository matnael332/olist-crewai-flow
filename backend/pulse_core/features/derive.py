"""Derive model features from checkout columns. Shared by the batch path and the Flask API so that
training and serving compute features identically."""

from __future__ import annotations

import pandas as pd

from pulse_core.features.spec import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


def derive_checkout_features(df: pd.DataFrame) -> pd.DataFrame:
    """df holds clean_data checkout columns (purchase_ts as datetime). Returns MODEL_FEATURES."""
    out = pd.DataFrame(index=df.index)
    passthrough = [
        "n_items",
        "n_distinct_products",
        "n_sellers",
        "total_price",
        "total_freight",
        "total_weight_g",
        "total_volume_cm3",
        "max_length_cm",
        "max_height_cm",
        "max_width_cm",
        "payment_total",
        "max_installments",
        "n_payments",
        "distance_km",
        "promised_days",
        "purchase_dow",
        "purchase_hour",
    ]
    for name in passthrough:
        out[name] = pd.to_numeric(df[name], errors="coerce").astype("float64")
    price = out["total_price"].clip(lower=0.01)
    out["freight_ratio"] = out["total_freight"] / price
    out["purchase_month_num"] = df["purchase_ts"].dt.month.astype("float64")
    out["is_weekend"] = (out["purchase_dow"] >= 5).astype("float64")
    out["same_state"] = df["same_state"].astype(bool).astype("float64")
    for name in CATEGORICAL_FEATURES:
        out[name] = df[name].astype("object")
    assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) == set(MODEL_FEATURES)
    return out[MODEL_FEATURES]
