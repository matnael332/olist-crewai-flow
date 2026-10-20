"""Run the cleaning rules end to end and write clean_data.csv + cleaning_report.json."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from olist_core.cleaning import rules
from olist_core.cleaning.geo import load_centroids
from olist_core.columns import CLEAN_COLUMNS, COLUMN_NAMES
from olist_core.io import write_csv_stable, write_json_lf
from olist_core.sorting import stable_sort

RENAMED = {
    "order_purchase_timestamp": "purchase_ts",
    "order_approved_at": "approved_ts",
    "order_delivered_carrier_date": "delivered_carrier_ts",
    "order_delivered_customer_date": "delivered_customer_ts",
    "order_estimated_delivery_date": "estimated_delivery_date",
}
DEFINITIONS = {
    "gmv": "sum of item prices in BRL, excluding freight",
    "aov": "GMV divided by the number of orders",
    "payment_total": "final charged amount; differs from price + freight by vouchers and interest",
    "is_late": "delivery_days > promised_days, both counted in calendar days from the purchase date",
}


def _cast(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COLUMN_NAMES].copy()
    for col in CLEAN_COLUMNS:
        if col.dtype == "int":
            out[col.name] = out[col.name].astype("Int64" if col.nullable else "int64")
        elif col.dtype == "float":
            out[col.name] = out[col.name].astype("float64")
        elif col.dtype == "bool":
            out[col.name] = out[col.name].astype("boolean" if col.nullable else "bool")
        elif col.dtype == "string":
            out[col.name] = out[col.name].astype("object")
    return out


def payment_reconciliation(df: pd.DataFrame) -> dict:
    diff = df["payment_total"] - (df["total_price"] + df["total_freight"])
    return {
        "orders": int(len(df)),
        "within_1_cent": int((diff.abs() <= 0.01).sum()),
        "paid_more_than_price_plus_freight": int((diff > 0.01).sum()),
        "paid_less_than_price_plus_freight": int((diff < -0.01).sum()),
        "median_difference_brl": round(float(diff.median()), 4),
        "p95_abs_difference_brl": round(float(np.percentile(diff.abs(), 95)), 4),
    }


def clean_olist(
    tables: dict[str, pd.DataFrame], centroids: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Apply R1-R10 to in-memory raw tables. Returns (clean_df, report)."""
    log = rules.CleaningLog()
    orders = tables["orders"]
    rows_in = len(orders)
    df = rules.r1_keep_delivered(orders, log)
    df = rules.r2_require_delivery_date(df, log)
    df = rules.r3_purchase_window(df, log)
    df = rules.r4_timestamp_sequences(df, log)
    df = rules.r5_items(
        df, tables["order_items"], tables["products"], tables["category_translation"], log
    )
    df = rules.r6_payments(df, tables["payments"], log)
    df = rules.r7_reviews(df, tables["reviews"], log)
    df = rules.r8_geography(df, tables["customers"], tables["sellers"], centroids, log)
    df = rules.r9_derive(df, log)
    df = df.rename(columns=RENAMED)
    before = len(df)
    df = stable_sort(_cast(df), ["purchase_ts", "order_id"])
    log.add(
        "R10_finalize",
        "fixed column order and dtypes, stable sort by (purchase_ts, order_id)",
        before,
        len(df),
        columns=len(COLUMN_NAMES),
    )
    report = {
        "rows_in": int(rows_in),
        "rows_out": int(len(df)),
        "rules": log.rules,
        "payment_reconciliation": payment_reconciliation(df),
        "definitions": DEFINITIONS,
    }
    return df, report


def run_cleaning(
    tables: dict[str, pd.DataFrame], lookups_dir: str | Path, out_dir: str | Path
) -> dict:
    """Clean and write clean_data.csv and cleaning_report.json into out_dir."""
    out_dir = Path(out_dir)
    clean, report = clean_olist(tables, load_centroids(lookups_dir))
    write_csv_stable(clean, out_dir / "clean_data.csv")
    write_json_lf(out_dir / "cleaning_report.json", report)
    return report
