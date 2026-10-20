"""Feature catalogue. Every feature declares the clean columns it is derived from; the features gate
checks that all of them are available at checkout (no leakage)."""

from __future__ import annotations

NUMERIC = "numeric"
CATEGORICAL = "categorical"


def _f(name, kind, derived_from, description, nullable=False):
    return {
        "name": name,
        "kind": kind,
        "derived_from": list(derived_from),
        "nullable": nullable,
        "description": description,
    }


FEATURE_DEFINITIONS: tuple[dict, ...] = (
    _f("n_items", NUMERIC, ["n_items"], "item units in the order"),
    _f("n_distinct_products", NUMERIC, ["n_distinct_products"], "distinct products"),
    _f(
        "n_sellers",
        NUMERIC,
        ["n_sellers"],
        "distinct sellers (multi-seller orders ship separately)",
    ),
    _f("total_price", NUMERIC, ["total_price"], "order GMV in BRL"),
    _f("total_freight", NUMERIC, ["total_freight"], "freight charged in BRL"),
    _f(
        "freight_ratio",
        NUMERIC,
        ["total_freight", "total_price"],
        "freight / price: expensive shipping relative to value signals distance or bulk",
    ),
    _f("total_weight_g", NUMERIC, ["total_weight_g"], "total weight", nullable=True),
    _f("total_volume_cm3", NUMERIC, ["total_volume_cm3"], "total package volume", nullable=True),
    _f("max_length_cm", NUMERIC, ["max_length_cm"], "largest package length", nullable=True),
    _f("max_height_cm", NUMERIC, ["max_height_cm"], "largest package height", nullable=True),
    _f("max_width_cm", NUMERIC, ["max_width_cm"], "largest package width", nullable=True),
    _f("payment_total", NUMERIC, ["payment_total"], "amount charged at checkout"),
    _f("max_installments", NUMERIC, ["max_installments"], "installments chosen"),
    _f("n_payments", NUMERIC, ["n_payments"], "payment records (vouchers split payments)"),
    _f("distance_km", NUMERIC, ["distance_km"], "customer-seller distance", nullable=True),
    _f("promised_days", NUMERIC, ["promised_days"], "delivery promise shown at checkout"),
    _f("purchase_month_num", NUMERIC, ["purchase_ts"], "month of year (seasonality)"),
    _f("purchase_dow", NUMERIC, ["purchase_dow"], "day of week, 0 = Monday"),
    _f("purchase_hour", NUMERIC, ["purchase_hour"], "hour of day"),
    _f("is_weekend", NUMERIC, ["purchase_dow"], "purchase on Saturday or Sunday"),
    _f("same_state", NUMERIC, ["same_state"], "customer and seller in the same state (0/1)"),
    _f("customer_state", CATEGORICAL, ["customer_state"], "customer state (UF)"),
    _f("seller_state", CATEGORICAL, ["seller_state"], "seller state (UF)"),
    _f("payment_type", CATEGORICAL, ["payment_type"], "type of the largest payment"),
    _f("primary_category", CATEGORICAL, ["primary_category"], "category of the first item"),
)

MODEL_FEATURES = [f["name"] for f in FEATURE_DEFINITIONS]
NUMERIC_FEATURES = [f["name"] for f in FEATURE_DEFINITIONS if f["kind"] == NUMERIC]
CATEGORICAL_FEATURES = [f["name"] for f in FEATURE_DEFINITIONS if f["kind"] == CATEGORICAL]
ID_COLUMNS = ["order_id", "purchase_ts", "split"]
LEAKAGE_RULE = (
    "every model feature is derived only from clean_data columns tagged available_at == "
    "'checkout'; identifiers and post-purchase columns are excluded"
)
