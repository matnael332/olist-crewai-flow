"""Validate checkout records against the dataset contract and turn them into model features.

Shared by the Flask API and the Streamlit Predict page. Uses the same derive_checkout_features as
training, so serving can never compute features differently from the model's training data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pulse_core.cleaning.geo import haversine_km
from pulse_core.features.derive import derive_checkout_features

MAX_RECORDS = 1000
REQUIRED = (
    "n_items",
    "total_price",
    "total_freight",
    "customer_state",
    "seller_state",
    "primary_category",
    "payment_type",
    "max_installments",
    "purchase_ts",
)
NUMERIC_OPTIONAL = (
    "n_distinct_products",
    "n_sellers",
    "total_weight_g",
    "total_volume_cm3",
    "max_length_cm",
    "max_height_cm",
    "max_width_cm",
    "payment_total",
    "n_payments",
    "distance_km",
    "promised_days",
)
STRING_OPTIONAL = ("estimated_delivery_date", "customer_zip_prefix", "seller_zip_prefix")
# Carried through to the response but never a model feature.
PASSTHROUGH = ("order_id",)
MAX_ORDER_ID = 64
ALLOWED_FIELDS = set(REQUIRED) | set(NUMERIC_OPTIONAL) | set(STRING_OPTIONAL) | set(PASSTHROUGH)
NUMERIC_FIELDS = {"n_items", "total_price", "total_freight", "max_installments", *NUMERIC_OPTIONAL}
DATETIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")


@dataclass
class ValidationResult:
    rows: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)


def _parse_dt(value) -> pd.Timestamp | None:
    if not isinstance(value, str):
        return None
    for fmt in DATETIME_FORMATS:
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue
    return None


class RecordValidator:
    def __init__(self, contract: dict, centroids: pd.DataFrame | None = None) -> None:
        self.columns = {c["name"]: c for c in contract["columns"]}
        self.centroids = centroids

    def _bounds(self, name: str) -> dict:
        return self.columns.get(name, {}).get("bounds", {})

    def _allowed(self, name: str) -> list | None:
        return self.columns.get(name, {}).get("allowed_values")

    def validate(self, records) -> ValidationResult:
        result = ValidationResult()
        if not isinstance(records, list) or not 1 <= len(records) <= MAX_RECORDS:
            result.errors.append(
                {
                    "index": None,
                    "field": "records",
                    "message": f"'records' must be a list of 1..{MAX_RECORDS} objects",
                }
            )
            return result
        for index, record in enumerate(records):
            row = self._validate_one(index, record, result)
            if row is not None:
                result.rows.append(row)
        return result

    def validate_partial(self, records: list) -> tuple[list[tuple[int, dict]], list, list]:
        """Validate each record on its own (batch scoring): valid (index, row) pairs, errors,
        warnings. Unlike validate(), one bad record does not reject the others."""
        result = ValidationResult()
        valid = []
        for index, record in enumerate(records):
            row = self._validate_one(index, record, result)
            if row is not None:
                valid.append((index, row))
        return valid, result.errors, result.warnings

    def _error(self, result, index, field_name, message) -> None:
        result.errors.append({"index": index, "field": field_name, "message": message})

    def _validate_one(self, index: int, record, result: ValidationResult) -> dict | None:
        if not isinstance(record, dict):
            self._error(result, index, None, "record must be a JSON object")
            return None
        before = len(result.errors)
        for name in sorted(set(record) - ALLOWED_FIELDS):
            self._error(result, index, name, "unknown field")
        order_id = record.get("order_id")
        if order_id is not None and (
            isinstance(order_id, bool)
            or not isinstance(order_id, str | int)
            or not 0 < len(str(order_id)) <= MAX_ORDER_ID
        ):
            self._error(
                result, index, "order_id", f"must be a string of 1..{MAX_ORDER_ID} characters"
            )
        for name in REQUIRED:
            if record.get(name) is None:
                self._error(result, index, name, "required")
        if record.get("estimated_delivery_date") is None and record.get("promised_days") is None:
            self._error(
                result, index, "promised_days", "provide estimated_delivery_date or promised_days"
            )
        for name in NUMERIC_FIELDS & set(record):
            value = record[name]
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                self._error(result, index, name, "must be a finite number")
                continue
            bounds = self._bounds(name)
            if "min" in bounds and value < bounds["min"]:
                self._error(result, index, name, f"must be >= {bounds['min']}")
            if "max" in bounds and value > bounds["max"]:
                self._error(result, index, name, f"must be <= {bounds['max']}")
        for name in ("customer_state", "seller_state", "primary_category", "payment_type"):
            value = record.get(name)
            allowed = self._allowed(name)
            if value is not None and (
                not isinstance(value, str) or (allowed and value not in allowed)
            ):
                self._error(result, index, name, f"{value!r} is not an allowed value")
        purchase = _parse_dt(record.get("purchase_ts"))
        if record.get("purchase_ts") is not None and purchase is None:
            self._error(result, index, "purchase_ts", "expected 'YYYY-MM-DD HH:MM:SS'")
        estimate = record.get("estimated_delivery_date")
        estimate_ts = _parse_dt(estimate) if estimate is not None else None
        if estimate is not None and estimate_ts is None:
            self._error(result, index, "estimated_delivery_date", "expected 'YYYY-MM-DD'")
        if len(result.errors) > before:
            return None

        promised = record.get("promised_days")
        if promised is None:
            promised = (estimate_ts.normalize() - purchase.normalize()).days
            if promised < 0:
                self._error(result, index, "estimated_delivery_date", "is before purchase_ts")
                return None
        distance = record.get("distance_km")
        cz, sz = record.get("customer_zip_prefix"), record.get("seller_zip_prefix")
        if distance is None and cz and sz and self.centroids is not None:
            c = self.centroids.reindex([str(cz).zfill(5)])
            s = self.centroids.reindex([str(sz).zfill(5)])
            distance = float(haversine_km(c["lat"], c["lng"], s["lat"], s["lng"])[0])
            if math.isnan(distance):
                result.warnings.append(
                    {
                        "index": index,
                        "field": "zip_prefix",
                        "message": "unknown zip prefix; distance treated as missing",
                    }
                )
        n_items = record["n_items"]
        return {
            "n_items": n_items,
            "n_distinct_products": record.get("n_distinct_products", n_items),
            "n_sellers": record.get("n_sellers", 1),
            "total_price": record["total_price"],
            "total_freight": record["total_freight"],
            "total_weight_g": record.get("total_weight_g", np.nan),
            "total_volume_cm3": record.get("total_volume_cm3", np.nan),
            "max_length_cm": record.get("max_length_cm", np.nan),
            "max_height_cm": record.get("max_height_cm", np.nan),
            "max_width_cm": record.get("max_width_cm", np.nan),
            "payment_total": record.get(
                "payment_total", record["total_price"] + record["total_freight"]
            ),
            "max_installments": record["max_installments"],
            "n_payments": record.get("n_payments", 1),
            "distance_km": np.nan if distance is None else distance,
            "promised_days": promised,
            "purchase_ts": purchase,
            "purchase_dow": purchase.dayofweek,
            "purchase_hour": purchase.hour,
            "same_state": record["customer_state"] == record["seller_state"],
            "customer_state": record["customer_state"],
            "seller_state": record["seller_state"],
            "payment_type": record["payment_type"],
            "primary_category": record["primary_category"],
        }


def to_features(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["purchase_ts"] = pd.to_datetime(frame["purchase_ts"])
    return derive_checkout_features(frame)
