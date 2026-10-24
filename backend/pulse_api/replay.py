"""Replay a past day: the delivered orders purchased on one test-window date, as API records.

The demo stand-in for "yesterday's orders". Records carry exactly the checkout-time values the
model was trained on (promised_days and distance_km included), so a replayed order scores the
same as its stored test prediction, and the real outcome (is_late) is available for a backtest.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from pulse_api.risk import LEVELS
from pulse_core.columns import read_clean_data
from pulse_core.policy import SPLITS

TEST_WINDOW = (date.fromisoformat(SPLITS["test"][0]), date.fromisoformat(SPLITS["test"][1]))
INT_FIELDS = ("n_items", "n_distinct_products", "n_sellers", "n_payments", "max_installments")
FLOAT_FIELDS = ("total_price", "total_freight", "payment_total")
OPTIONAL_FLOATS = (
    "total_weight_g",
    "total_volume_cm3",
    "max_length_cm",
    "max_height_cm",
    "max_width_cm",
    "distance_km",
)
TEXT_FIELDS = ("customer_state", "seller_state", "primary_category", "payment_type")


class ReplayError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message

    def to_dict(self) -> dict:
        return {"error": self.code, "detail": self.message}


def parse_day(value: str | None) -> date:
    """'YYYY-MM-DD' inside the test window, or a ReplayError (400 / 404)."""
    try:
        day = date.fromisoformat(str(value))
    except ValueError:
        raise ReplayError(400, "invalid_date", "date must be YYYY-MM-DD") from None
    start, end = TEST_WINDOW
    if not start <= day <= end:
        raise ReplayError(
            404, "out_of_window", f"replay is available for {start} to {end} (the test window)"
        )
    return day


@lru_cache(maxsize=2)
def _clean(path: str, mtime: float) -> pd.DataFrame:
    return read_clean_data(path)


def load_clean(artifacts_dir: Path) -> pd.DataFrame:
    path = Path(artifacts_dir) / "clean_data.csv"
    return _clean(str(path), path.stat().st_mtime)


def record_from_clean_row(row) -> dict:
    """One clean_data row -> the /predict record the model was trained on."""
    record = {"order_id": row.order_id}
    record.update({name: int(getattr(row, name)) for name in INT_FIELDS})
    record.update({name: float(getattr(row, name)) for name in FLOAT_FIELDS})
    record.update({name: getattr(row, name) for name in TEXT_FIELDS})
    record["purchase_ts"] = row.purchase_ts.strftime("%Y-%m-%d %H:%M:%S")
    record["promised_days"] = int(row.promised_days)
    for name in OPTIONAL_FLOATS:
        value = getattr(row, name)
        if pd.notna(value):
            record[name] = float(value)
    return record


def orders_for_day(clean: pd.DataFrame, day: date) -> tuple[list[dict], dict[str, int]]:
    """(records, actual outcome by order_id) for the orders purchased on ``day``."""
    rows = clean.loc[clean["purchase_ts"].dt.date == day]
    rows = rows.sort_values(["purchase_ts", "order_id"], kind="mergesort")
    if rows.empty:
        raise ReplayError(404, "no_orders", f"no delivered orders were purchased on {day}")
    records = [record_from_clean_row(row) for row in rows.itertuples(index=False)]
    outcomes = dict(zip(rows["order_id"], rows["is_late"].astype(int), strict=True))
    return records, outcomes


def backtest(predictions: list[dict], outcomes: dict[str, int]) -> dict:
    """How many orders of each level were actually late (only possible for a replayed day)."""
    out = {}
    for level in LEVELS:
        rows = [p for p in predictions if p["risk_level"] == level]
        late = sum(outcomes.get(p["order_id"], 0) for p in rows)
        out[level] = {
            "orders": len(rows),
            "late": late,
            "late_rate": round(late / len(rows), 4) if rows else None,
        }
    total = len(outcomes)
    out["all"] = {
        "orders": total,
        "late": sum(outcomes.values()),
        "late_rate": round(sum(outcomes.values()) / total, 4) if total else None,
    }
    return out
