"""Project-wide policy constants shared by the contract, features, EDA and training.

Keeping them in one place guarantees that the dataset contract, the feature gate and the trainer
all agree on the target definition, the time split and the required/forbidden features.
"""

from __future__ import annotations

import pandas as pd

SEED = 42
SNAPSHOT_DATE = "2018-10-17"  # last purchase timestamp in the raw data
CONTRACT_VERSION = "1.0"

TARGET = "is_late"
TARGET_DEFINITION = (
    "delivery_days > promised_days, both counted in calendar days from the purchase date "
    "(delivered on the promised date counts as on time); measured among delivered orders only"
)
REJECTED_TARGET = "timestamp comparison delivered_ts > estimated_date (8.1% positive)"

SPLITS = {
    "train": ("2017-01-01", "2018-04-30"),
    "test": ("2018-05-01", "2018-07-31"),
    "censored_buffer": ("2018-08-01", "2018-08-31"),
}
SPLIT_POLICY = {
    "type": "time",
    "column": "purchase_ts",
    "train": list(SPLITS["train"]),
    "test": list(SPLITS["test"]),
    "censored_buffer": list(SPLITS["censored_buffer"]),
    "cv": "TimeSeriesSplit(n_splits=3) on the train split ordered by purchase_ts",
    "note": (
        "August 2018 purchases are excluded from metrics: the snapshot ends 2018-10-17, so very "
        "late August orders may still have been undelivered (right-censoring)"
    ),
}

# Required features: what the Data Scientist crew must build (derived only from checkout columns).
REQUIRED_FEATURES = (
    {"name": "n_items", "derived_from": ["n_items"]},
    {"name": "total_price", "derived_from": ["total_price"]},
    {"name": "total_freight", "derived_from": ["total_freight"]},
    {"name": "total_weight_g", "derived_from": ["total_weight_g"]},
    {"name": "total_volume_cm3", "derived_from": ["total_volume_cm3"]},
    {"name": "customer_state", "derived_from": ["customer_state"]},
    {"name": "seller_state", "derived_from": ["seller_state"]},
    {"name": "same_state", "derived_from": ["same_state"]},
    {"name": "primary_category", "derived_from": ["primary_category"]},
    {"name": "payment_type", "derived_from": ["payment_type"]},
    {"name": "max_installments", "derived_from": ["max_installments"]},
    {"name": "promised_days", "derived_from": ["promised_days"]},
    {"name": "purchase_month_num", "derived_from": ["purchase_ts"]},
    {"name": "purchase_dow", "derived_from": ["purchase_dow"]},
)
FORBIDDEN_FEATURES = (
    "approved_ts",
    "delivered_carrier_ts",
    "delivered_customer_ts",
    "delivery_days",
    "review_score",
    "bad_review",
    "is_late",
)

BRAZIL_STATES = (
    "AC AL AM AP BA CE DF ES GO MA MG MS MT PA PB PE PI PR RJ RN RO RR RS SC SE SP TO".split()
)
PAYMENT_TYPES = ("boleto", "credit_card", "debit_card", "voucher")


def assign_split(purchase_ts: pd.Series) -> pd.Series:
    """Map purchase timestamps to train / test / censored_buffer (outside -> "excluded")."""
    day = purchase_ts.dt.strftime("%Y-%m-%d")
    split = pd.Series("excluded", index=purchase_ts.index, dtype="object")
    for name, (start, end) in SPLITS.items():
        split[(day >= start) & (day <= end)] = name
    return split
