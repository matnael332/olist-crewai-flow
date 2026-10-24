"""Batch scoring: a CSV (or a list of records) in, a ranked risk list out.

Unlike POST /predict, one bad row does not reject the batch: valid rows are scored, bad rows are
reported with their row number and field. Shared by POST /predict/batch, GET /risk-list, the ops
copilot and the dashboard.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pulse_api.model import ModelBundle
from pulse_api.risk import LEVELS
from pulse_api.validation import ALLOWED_FIELDS, NUMERIC_FIELDS, REQUIRED

MAX_BATCH_ROWS = 5000
OUTPUT_COLUMNS = [
    "row",
    "order_id",
    "customer_state",
    "late_probability",
    "risk_level",
    "recommended_action",
    "customer_message",
]


class BatchError(Exception):
    """A problem with the whole file (not a single row): maps to one HTTP error."""

    def __init__(self, status: int, code: str, message: str, **details) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details

    def to_dict(self) -> dict:
        return {"error": self.code, "detail": self.message, **self.details}


@dataclass
class BatchResult:
    predictions: list[dict] = field(default_factory=list)  # highest risk first
    errors: list[dict] = field(default_factory=list)  # {row, field, message}
    warnings: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        extra = [c for c in ("actual_late",) if self.predictions and c in self.predictions[0]]
        return pd.DataFrame(self.predictions, columns=OUTPUT_COLUMNS + extra)

    def to_csv(self) -> str:
        return self.to_frame().to_csv(index=False, lineterminator="\n")

    def action_list(self) -> pd.DataFrame:
        """High and elevated orders only: the operator's to-do list."""
        frame = self.to_frame()
        return frame[frame["risk_level"].isin(["high", "elevated"])].reset_index(drop=True)


def read_orders_csv(data: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV of orders (all values read as text; types are checked per row)."""
    if not data or not data.strip():
        raise BatchError(400, "empty", "the file is empty")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise BatchError(400, "not_csv", "the file is not UTF-8 text") from None
    try:
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise BatchError(400, "not_csv", f"cannot parse the CSV: {exc}") from None
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.empty:
        raise BatchError(400, "empty", "the file has a header but no rows")
    missing = [c for c in REQUIRED if c not in frame.columns]
    if "estimated_delivery_date" not in frame.columns and "promised_days" not in frame.columns:
        missing.append("estimated_delivery_date or promised_days")
    if missing:
        raise BatchError(400, "missing_columns", "required columns are missing", missing=missing)
    if len(frame) > MAX_BATCH_ROWS:
        raise BatchError(
            413, "too_large", f"at most {MAX_BATCH_ROWS:,} rows per batch", rows=len(frame)
        )
    return frame


def _value(name: str, value):
    if isinstance(value, np.generic):
        value = value.item()
    if name in NUMERIC_FIELDS and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value  # the validator reports "must be a finite number" for this row
    if isinstance(value, str):
        return value.strip()
    return value


def frame_to_records(frame: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """Rows -> API records (empty cells omitted); returns (records, ignored column names)."""
    ignored = sorted(set(frame.columns) - ALLOWED_FIELDS)
    columns = [c for c in frame.columns if c in ALLOWED_FIELDS]
    records = []
    for values in frame[columns].itertuples(index=False, name=None):
        record = {}
        for name, value in zip(columns, values, strict=True):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            if isinstance(value, str) and not value.strip():
                continue
            record[name] = _value(name, value)
        records.append(record)
    return records, ignored


def score_records(
    bundle: ModelBundle, records: list[dict], ignored_columns: list[str] | None = None
) -> BatchResult:
    """Validate each record, score the valid ones, rank by risk (row numbers start at 1)."""
    result = BatchResult()
    valid, errors, warnings = bundle.validator.validate_partial(records)
    result.errors = [
        {"row": e["index"] + 1, "field": e["field"], "message": e["message"]} for e in errors
    ]
    result.warnings = [
        {"row": w["index"] + 1, "field": w["field"], "message": w["message"]} for w in warnings
    ]
    if ignored_columns:
        result.warnings.insert(
            0,
            {
                "row": None,
                "field": ", ".join(ignored_columns),
                "message": "ignored columns (not part of the API schema)",
            },
        )
    if valid:
        probabilities = bundle.score_rows([row for _, row in valid])
        for (index, _), probability in zip(valid, probabilities, strict=True):
            record = records[index]
            scored = bundle.describe(index, float(probability), record)
            level = scored["risk_level"]
            result.predictions.append(
                {
                    "row": index + 1,
                    "order_id": scored.get("order_id", ""),
                    "customer_state": record.get("customer_state", ""),
                    "late_probability": scored["late_probability"],
                    "risk_level": level,
                    "recommended_action": scored["recommended_action"],
                    "customer_message": bundle.policy.customer_message(
                        level, scored.get("order_id"), record.get("customer_state")
                    ),
                }
            )
        result.predictions.sort(key=lambda r: (-r["late_probability"], r["row"]))
    rejected = {e["row"] for e in result.errors}
    result.summary = {
        "rows": len(records),
        "scored": len(result.predictions),
        "rejected": len(rejected),
        **{level: sum(r["risk_level"] == level for r in result.predictions) for level in LEVELS},
    }
    # No "expected late orders" (sum of probabilities): the model over-predicts on recent months
    # (trained at a 7.6% late rate, tested at 3.8%), so a sum would mislead. Levels rank; they
    # do not count.
    return result


def score_csv(bundle: ModelBundle, data: bytes) -> BatchResult:
    frame = read_orders_csv(data)
    records, ignored = frame_to_records(frame)
    return score_records(bundle, records, ignored)
