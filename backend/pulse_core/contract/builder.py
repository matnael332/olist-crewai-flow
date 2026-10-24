"""Build dataset_contract.json from the written clean_data.csv (round trip).

The contract is derived from the file exactly as the gate will read it back, so observed values can
never disagree with what the gate sees. Binding rules (bounds, enums, patterns, constraints) are
declared, not observed; observed ranges are informational only. No run id or timestamp is stored,
so the contract is byte-identical across reproducible runs.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from pulse_core.columns import CLEAN_COLUMNS, read_clean_data
from pulse_core.narratives import ContractNarrative
from pulse_core.policy import (
    BRAZIL_STATES,
    CONTRACT_VERSION,
    FORBIDDEN_FEATURES,
    PAYMENT_TYPES,
    REJECTED_TARGET,
    REQUIRED_FEATURES,
    SNAPSHOT_DATE,
    SPLIT_POLICY,
    TARGET,
    TARGET_DEFINITION,
    assign_split,
)

PATTERNS = {
    "order_id": r"^[0-9a-f]{32}$",
    "customer_unique_id": r"^[0-9a-f]{32}$",
    "primary_product_id": r"^[0-9a-f]{32}$",
    "primary_seller_id": r"^[0-9a-f]{32}$",
    "customer_zip_prefix": r"^[0-9]{5}$",
    "seller_zip_prefix": r"^[0-9]{5}$",
    "purchase_month": r"^20[0-9]{2}-[01][0-9]$",
}
BOUNDS = {
    "purchase_dow": {"min": 0, "max": 6},
    "purchase_hour": {"min": 0, "max": 23},
    "n_items": {"min": 1},
    "n_distinct_products": {"min": 1},
    "n_sellers": {"min": 1},
    "n_payments": {"min": 1},
    "total_price": {"min": 0},
    "total_freight": {"min": 0},
    "payment_total": {"min": 0},
    "total_weight_g": {"min": 0},
    "total_volume_cm3": {"min": 0},
    "max_length_cm": {"min": 0},
    "max_height_cm": {"min": 0},
    "max_width_cm": {"min": 0},
    "distance_km": {"min": 0},
    "max_installments": {"min": 0},
    "promised_days": {"min": 0, "max": 365},
    "delivery_days": {"min": 0, "max": 365},
    "review_score": {"min": 1, "max": 5},
}
CONSTRAINTS = [
    {"id": "C01", "type": "unique", "columns": ["order_id"], "description": "one row per order"},
    {
        "id": "C02",
        "type": "query",
        "query": "n_items >= n_distinct_products and n_distinct_products >= 1 and n_sellers >= 1",
        "description": "item counts are consistent",
    },
    {
        "id": "C03",
        "type": "query",
        "query": "total_price >= 0 and total_freight >= 0 and payment_total >= 0",
        "description": "amounts are non-negative",
    },
    {
        "id": "C04",
        "type": "query",
        "query": "max_installments >= 0 and n_payments >= 1",
        "description": "every order has a valid payment",
    },
    {
        "id": "C05",
        "type": "query",
        "query": "promised_days >= 0 and delivery_days >= 0",
        "description": "day counts are non-negative",
    },
    {
        "id": "C06",
        "type": "query",
        "query": "is_late == (delivery_days > promised_days)",
        "description": "target definition",
    },
    {
        "id": "C07",
        "type": "query",
        "query": "same_state == (customer_state == seller_state)",
        "description": "same_state is consistent",
    },
    {
        "id": "C08",
        "type": "query",
        "query": "delivered_customer_ts >= purchase_ts",
        "description": "delivery happens after purchase",
    },
    {
        "id": "C09",
        "type": "query",
        "query": "review_score >= 1 and review_score <= 5",
        "where_not_null": ["review_score"],
        "description": "review score range",
    },
    {
        "id": "C10",
        "type": "query",
        "query": "bad_review == (review_score <= 2)",
        "where_not_null": ["review_score", "bad_review"],
        "description": "bad_review consistency",
    },
    {
        "id": "C11",
        "type": "date_range",
        "column": "purchase_ts",
        "min": "2017-01-01 00:00:00",
        "max_exclusive": "2018-09-01 00:00:00",
        "description": "purchase window",
    },
]


def _observed(series: pd.Series, dtype: str) -> dict | None:
    values = series.dropna()
    if values.empty or dtype not in ("int", "float", "datetime"):
        return None
    lo, hi = values.min(), values.max()
    if dtype == "datetime":
        return {"min": lo.strftime("%Y-%m-%d %H:%M:%S"), "max": hi.strftime("%Y-%m-%d %H:%M:%S")}
    if dtype == "float":  # widened outward at 4 decimals: informational, never binding
        return {"min": math.floor(float(lo) * 1e4) / 1e4, "max": math.ceil(float(hi) * 1e4) / 1e4}
    return {"min": int(lo), "max": int(hi)}


def _allowed_values(name: str, series: pd.Series) -> list | None:
    if name in ("customer_state", "seller_state"):
        return list(BRAZIL_STATES)
    if name == "payment_type":
        return list(PAYMENT_TYPES)
    if name == "primary_category":
        return sorted(series.dropna().unique().tolist())
    if name in ("same_state", "is_late", "bad_review"):
        return [False, True]
    return None


def merge_narrative(base: ContractNarrative, extra: ContractNarrative | None) -> dict:
    """Template items always stay; LLM items are appended when new (case-insensitive)."""
    known = {c.name for c in CLEAN_COLUMNS}

    def merge(a: list[str], b: list[str]) -> list[str]:
        seen = {x.strip().lower() for x in a}
        return a + [x for x in b if x.strip() and x.strip().lower() not in seen]

    notes = [n.model_dump() for n in base.column_notes]
    if extra is not None:
        noted = {(n["name"], n["note"].strip().lower()) for n in notes}
        notes += [
            n.model_dump()
            for n in extra.column_notes
            if n.name in known and (n.name, n.note.strip().lower()) not in noted
        ]
    source = "template+llm" if extra is not None else "template"
    return {
        "assumptions": {
            "narrative_source": source,
            "items": merge(base.assumptions, extra.assumptions if extra else []),
        },
        "business_rules": {
            "narrative_source": source,
            "items": merge(base.business_rules, extra.business_rules if extra else []),
        },
        "column_notes": {"narrative_source": source, "items": notes},
        "known_limitations": {
            "narrative_source": source,
            "items": merge(base.known_limitations, extra.known_limitations if extra else []),
        },
    }


def build_contract(
    clean_csv: str | Path,
    cleaning_report: dict,
    provenance: dict,
    narrative: dict,
) -> dict:
    clean = read_clean_data(clean_csv)
    columns = []
    for col in CLEAN_COLUMNS:
        series = clean[col.name]
        entry = {
            "name": col.name,
            "dtype": col.dtype,
            "nullable": col.nullable,
            "available_at": col.available_at,
            "description": col.description,
        }
        if col.name == "order_id":
            entry["unique"] = True
        if col.name in PATTERNS:
            entry["pattern"] = PATTERNS[col.name]
        if col.name in BOUNDS:
            entry["bounds"] = BOUNDS[col.name]
        allowed = _allowed_values(col.name, series)
        if allowed is not None:
            entry["allowed_values"] = allowed
        observed = _observed(series, col.dtype)
        if observed is not None:
            entry["observed_range"] = observed
        columns.append(entry)

    split = assign_split(clean["purchase_ts"])
    late = clean[TARGET].astype(bool)
    rates = {"overall": float(late.mean())}
    for name in ("train", "test", "censored_buffer"):
        rates[name] = float(late[split == name].mean())

    return {
        "contract_version": CONTRACT_VERSION,
        "dataset": {
            "name": "olist_delivered_orders_clean",
            "source": "Olist Brazilian E-Commerce Public Dataset (GitHub olist/work-at-olist-data)",
            "license": "MIT (GitHub repository); Kaggle listing CC BY-NC-SA 4.0. Attribute Olist; "
            "non-commercial coursework.",
            "grain": "one row per delivered order",
            "currency": "BRL",
            "snapshot_date": SNAPSHOT_DATE,
        },
        "provenance": {
            **provenance,
            "cleaning_rules": [
                {"rule": r["rule"], "rows_removed": r["rows_removed"]}
                for r in cleaning_report["rules"]
            ],
        },
        "row_count": int(len(clean)),
        "primary_key": ["order_id"],
        "date_range": {
            "column": "purchase_ts",
            "min": clean["purchase_ts"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "max": clean["purchase_ts"].max().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "columns": columns,
        "constraints": CONSTRAINTS,
        "population": {
            "order_status": "delivered",
            "purchase_window": ["2017-01-01", "2018-08-31"],
            "excluded": "non-delivered orders, delivered orders without a delivery date, orders "
            "without items or a valid payment, orders delivered before purchase",
        },
        "target": {
            "name": TARGET,
            "task": "binary_classification",
            "entity": "order",
            "decision_time": "checkout (purchase_ts)",
            "definition": TARGET_DEFINITION,
            "rejected_alternative": REJECTED_TARGET,
            "expected_positive_rate": rates,
            "split_policy": SPLIT_POLICY,
        },
        "required_features": [{**feat, "available_at": "checkout"} for feat in REQUIRED_FEATURES],
        "forbidden_features": list(FORBIDDEN_FEATURES),
        "feature_rules": {
            "allowed_sources": "only columns with available_at == 'checkout'",
            "identifiers_excluded": [
                c.name for c in CLEAN_COLUMNS if c.available_at == "identifier"
            ],
            "leakage_rule": "a feature may only be derived from information known at purchase time",
        },
        **narrative,
        "privacy": {
            "kept": ["customer_unique_id", "customer_zip_prefix", "seller_zip_prefix"],
            "dropped": ["customer_city", "seller_city", "review text"],
            "note": "Identifiers are Olist's public pseudonymous hashes; zip prefix plus timestamps "
            "are quasi-identifiers, so dashboards show aggregates only.",
        },
    }
