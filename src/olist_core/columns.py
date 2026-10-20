"""The clean_data.csv column catalogue: order, logical dtype, availability and meaning.

available_at drives the leakage rules: only "checkout" columns may feed model features;
"post_purchase" columns are forbidden; "identifier" columns are never features; "label" is the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # string | int | float | bool | datetime
    available_at: str  # identifier | checkout | post_purchase | label
    nullable: bool
    description: str


CLEAN_COLUMNS: tuple[Column, ...] = (
    Column("order_id", "string", "identifier", False, "Olist order id (32 hex chars); primary key"),
    Column(
        "customer_unique_id",
        "string",
        "identifier",
        False,
        "Pseudonymous id of the person (stable across their orders)",
    ),
    Column("primary_product_id", "string", "identifier", False, "Product of the first order item"),
    Column("primary_seller_id", "string", "identifier", False, "Seller of the first order item"),
    Column(
        "customer_zip_prefix",
        "string",
        "identifier",
        False,
        "First 5 digits of the customer's postcode",
    ),
    Column(
        "seller_zip_prefix",
        "string",
        "identifier",
        False,
        "First 5 digits of the primary seller's postcode",
    ),
    Column("purchase_ts", "datetime", "checkout", False, "Purchase timestamp (decision time)"),
    Column("purchase_month", "string", "checkout", False, "Purchase month, YYYY-MM"),
    Column("purchase_dow", "int", "checkout", False, "Purchase day of week, 0 = Monday"),
    Column("purchase_hour", "int", "checkout", False, "Purchase hour of day, 0-23"),
    Column("n_items", "int", "checkout", False, "Number of item units in the order"),
    Column("n_distinct_products", "int", "checkout", False, "Number of distinct products"),
    Column("n_sellers", "int", "checkout", False, "Number of distinct sellers"),
    Column(
        "total_price",
        "float",
        "checkout",
        False,
        "Order GMV: sum of item prices in BRL, excluding freight",
    ),
    Column("total_freight", "float", "checkout", False, "Sum of freight charges in BRL"),
    Column(
        "total_weight_g",
        "float",
        "checkout",
        True,
        "Sum of item weights in grams (null if any product lacks dimensions)",
    ),
    Column("total_volume_cm3", "float", "checkout", True, "Sum of item package volumes in cm3"),
    Column("max_length_cm", "float", "checkout", True, "Largest item package length"),
    Column("max_height_cm", "float", "checkout", True, "Largest item package height"),
    Column("max_width_cm", "float", "checkout", True, "Largest item package width"),
    Column(
        "primary_category",
        "string",
        "checkout",
        False,
        "English category of the first item ('unknown' when missing)",
    ),
    Column(
        "payment_total",
        "float",
        "checkout",
        False,
        "Final charged amount in BRL (includes vouchers and installment interest)",
    ),
    Column("payment_type", "string", "checkout", False, "Type of the largest payment"),
    Column("max_installments", "int", "checkout", False, "Largest number of installments"),
    Column("n_payments", "int", "checkout", False, "Number of payment records"),
    Column("customer_state", "string", "checkout", False, "Customer state (UF code)"),
    Column("seller_state", "string", "checkout", False, "Primary seller state (UF code)"),
    Column("same_state", "bool", "checkout", False, "Customer and seller in the same state"),
    Column(
        "distance_km",
        "float",
        "checkout",
        True,
        "Great-circle distance between customer and seller zip-prefix centroids",
    ),
    Column(
        "estimated_delivery_date",
        "datetime",
        "checkout",
        False,
        "Delivery estimate shown to the customer at purchase",
    ),
    Column(
        "promised_days",
        "int",
        "checkout",
        False,
        "Calendar days from purchase date to the estimated delivery date",
    ),
    Column("approved_ts", "datetime", "post_purchase", True, "Payment approval timestamp"),
    Column(
        "delivered_carrier_ts",
        "datetime",
        "post_purchase",
        True,
        "Hand-over to the carrier (nulled when out of sequence)",
    ),
    Column("delivered_customer_ts", "datetime", "post_purchase", False, "Delivery to the customer"),
    Column(
        "delivery_days",
        "int",
        "post_purchase",
        False,
        "Calendar days from purchase date to delivery date",
    ),
    Column("review_score", "int", "post_purchase", True, "Latest review score, 1-5"),
    Column("bad_review", "bool", "post_purchase", True, "Latest review score <= 2"),
    Column("is_late", "bool", "label", False, "Delivered after the promised date (calendar days)"),
)

COLUMN_NAMES = [c.name for c in CLEAN_COLUMNS]
COLUMNS_BY_NAME = {c.name: c for c in CLEAN_COLUMNS}
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def columns_available_at(stage: str) -> list[str]:
    return [c.name for c in CLEAN_COLUMNS if c.available_at == stage]


def parse_column(series: pd.Series, dtype: str) -> tuple[pd.Series, int]:
    """Convert a string column to its logical dtype. Returns (typed, unparseable non-empty count).

    Integers and booleans use pandas nullable dtypes so that null violations stay visible.
    """
    empty = series.isna() | (series == "")
    if dtype == "string":
        return series.where(~empty, None).astype("object"), 0
    if dtype in ("int", "float"):
        numbers = pd.to_numeric(series.where(~empty), errors="coerce")
        bad = numbers.isna() & ~empty
        if dtype == "int":
            non_integral = numbers.notna() & (numbers != numbers.round())
            bad = bad | non_integral
            numbers = numbers.where(~non_integral)
            return numbers.round().astype("Int64"), int(bad.sum())
        return numbers.astype("float64"), int(bad.sum())
    if dtype == "bool":
        mapped = series.map({"True": True, "False": False})
        bad = mapped.isna() & ~empty
        return mapped.astype("boolean"), int(bad.sum())
    if dtype == "datetime":
        parsed = pd.to_datetime(series.where(~empty), format=DATETIME_FORMAT, errors="coerce")
        return parsed, int((parsed.isna() & ~empty).sum())
    raise ValueError(f"unknown logical dtype {dtype!r}")


def read_csv_strings(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def read_clean_data(path, dtypes: dict[str, str] | None = None) -> pd.DataFrame:
    """Read clean_data.csv with explicit types (zip prefixes keep leading zeros).

    dtypes defaults to the catalogue; the contract gate passes the contract's own dtypes.
    """
    raw = read_csv_strings(path)
    dtypes = dtypes or {c.name: c.dtype for c in CLEAN_COLUMNS}
    typed = {}
    for name in raw.columns:
        typed[name], _ = parse_column(raw[name], dtypes.get(name, "string"))
    return pd.DataFrame(typed, index=raw.index)
