"""Record validation for the prediction API, driven by feature_spec.json and the contract.

A record carries the model features themselves (the fields listed in ``model_features``), plus an
optional identifier echoed back. Every feature is checked by its kind (numeric, categorical), by
nullability, and, when the feature is a pass-through of a contract column, by that column's
bounds and allowed values. A dataset spec may later add a raw-record adapter; until then clients
send already-derived features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

MAX_RECORDS = 1000
MAX_ID_LENGTH = 64


@dataclass
class ValidationResult:
    rows: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)  # {index, field, message}
    warnings: list[dict] = field(default_factory=list)


class RecordValidator:
    def __init__(self, contract: dict, feature_spec: dict) -> None:
        self.id_field = feature_spec.get("id_column")
        self.features = list(feature_spec["model_features"])
        definitions = {f["name"]: f for f in feature_spec["features"]}
        self.kinds = {name: definitions[name]["kind"] for name in self.features}
        self.nullable = {name: bool(definitions[name].get("nullable")) for name in self.features}
        columns = {c["name"]: c for c in contract.get("columns", [])}
        self.rules = {}
        for name in self.features:
            sources = definitions[name].get("derived_from", [])
            if sources == [name] and name in columns:
                self.rules[name] = columns[name]
        self.required_fields = [n for n in self.features if not self.nullable[n]]
        self.numeric_fields = {n for n, k in self.kinds.items() if k == "numeric"}
        self.allowed_fields = set(self.features) | ({self.id_field} if self.id_field else set())

    # ------------------------------------------------------------------ single record
    def _check(self, index: int, record, errors: list[dict]) -> dict | None:
        if not isinstance(record, dict):
            errors.append({"index": index, "field": None, "message": "record must be an object"})
            return None
        row: dict = {}
        ok = True
        for name in record:
            if name not in self.allowed_fields:
                errors.append({"index": index, "field": name, "message": "unknown field"})
                ok = False
        for name in self.features:
            value = record.get(name)
            if value is None or (isinstance(value, str) and not value.strip()):
                if self.nullable[name]:
                    row[name] = None
                    continue
                errors.append({"index": index, "field": name, "message": "required"})
                ok = False
                continue
            if self.kinds[name] == "numeric":
                if isinstance(value, bool):
                    value = float(value)
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = math.nan
                if not math.isfinite(number):
                    errors.append(
                        {"index": index, "field": name, "message": "must be a finite number"}
                    )
                    ok = False
                    continue
                bounds = self.rules.get(name, {}).get("bounds") or {}
                if "min" in bounds and number < bounds["min"]:
                    errors.append(
                        {"index": index, "field": name, "message": f"below {bounds['min']}"}
                    )
                    ok = False
                if "max" in bounds and number > bounds["max"]:
                    errors.append(
                        {"index": index, "field": name, "message": f"above {bounds['max']}"}
                    )
                    ok = False
                row[name] = number
            else:
                text = str(value).strip()
                allowed = self.rules.get(name, {}).get("allowed_values")
                if allowed is not None and text not in {str(v) for v in allowed}:
                    errors.append(
                        {"index": index, "field": name, "message": f"{text!r} is not allowed"}
                    )
                    ok = False
                row[name] = text
        if self.id_field and record.get(self.id_field) is not None:
            if len(str(record[self.id_field])) > MAX_ID_LENGTH:
                errors.append(
                    {"index": index, "field": self.id_field, "message": "identifier too long"}
                )
                ok = False
        return row if ok else None

    # ------------------------------------------------------------------ batches
    def validate(self, records) -> ValidationResult:
        """All-or-nothing: any error empties ``rows``."""
        result = ValidationResult()
        if not isinstance(records, list) or not records:
            result.errors.append(
                {"index": None, "field": None, "message": "records must be a non-empty list"}
            )
            return result
        if len(records) > MAX_RECORDS:
            result.errors.append(
                {"index": None, "field": None, "message": f"at most {MAX_RECORDS} records"}
            )
            return result
        rows = [self._check(i, r, result.errors) for i, r in enumerate(records)]
        result.rows = [] if result.errors else [r for r in rows if r is not None]
        return result

    def validate_partial(self, records) -> tuple[list[tuple[int, dict]], list[dict], list[dict]]:
        """Partial success: (valid [(index, row)], errors, warnings)."""
        errors: list[dict] = []
        valid = []
        for i, record in enumerate(records):
            row = self._check(i, record, errors)
            if row is not None:
                valid.append((i, row))
        return valid, errors, []

    def to_features(self, rows: list[dict]) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=self.features)
        for name, kind in self.kinds.items():
            frame[name] = (
                frame[name].astype("float64") if kind == "numeric" else frame[name].astype(object)
            )
        return frame
