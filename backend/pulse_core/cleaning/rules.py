"""Cleaning rules R1-R10. Each rule is a pure function that records what it changed.

Tables have different grains (order, item, payment, review), so items, payments and reviews are
aggregated to one row per order before they are joined; a naive join would double-count revenue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pulse_core.cleaning.geo import haversine_km
from pulse_core.sorting import stable_sort

WINDOW_START = pd.Timestamp("2017-01-01")
WINDOW_END_EXCLUSIVE = pd.Timestamp("2018-09-01")
UNKNOWN_CATEGORY = "unknown"


@dataclass
class CleaningLog:
    rules: list[dict] = field(default_factory=list)

    def add(self, rule: str, description: str, before: int, after: int, **detail) -> None:
        self.rules.append(
            {
                "rule": rule,
                "description": description,
                "rows_before": int(before),
                "rows_after": int(after),
                "rows_removed": int(before - after),
                "detail": {k: v for k, v in sorted(detail.items())},
            }
        )


def r1_keep_delivered(orders: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    removed = orders.loc[orders["order_status"] != "delivered", "order_status"]
    by_status = {str(k): int(v) for k, v in sorted(removed.value_counts(sort=False).items())}
    kept = orders[orders["order_status"] == "delivered"]
    log.add(
        "R1_keep_delivered",
        "keep orders with status 'delivered' (lateness is only defined for delivered orders)",
        len(orders),
        len(kept),
        removed_by_status=by_status,
    )
    return kept


def r2_require_delivery_date(orders: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    kept = orders[orders["order_delivered_customer_date"].notna()]
    log.add(
        "R2_require_delivery_date",
        "drop 'delivered' orders that have no customer delivery date",
        len(orders),
        len(kept),
    )
    return kept


def r3_purchase_window(orders: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    ts = orders["order_purchase_timestamp"]
    kept = orders[(ts >= WINDOW_START) & (ts < WINDOW_END_EXCLUSIVE)]
    log.add(
        "R3_purchase_window",
        "keep purchases from 2017-01-01 to 2018-08-31 (2016 and Sep/Oct 2018 are sparse edges)",
        len(orders),
        len(kept),
        window=["2017-01-01", "2018-08-31"],
    )
    return kept


def r4_timestamp_sequences(orders: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    orders = orders.copy()
    purchase = orders["order_purchase_timestamp"]
    delivered = orders["order_delivered_customer_date"]
    carrier = orders["order_delivered_carrier_date"]
    impossible = delivered < purchase
    kept = orders[~impossible].copy()
    carrier = kept["order_delivered_carrier_date"]
    before_purchase = carrier < kept["order_purchase_timestamp"]
    after_delivery = kept["order_delivered_customer_date"] < carrier
    kept.loc[before_purchase | after_delivery, "order_delivered_carrier_date"] = pd.NaT
    log.add(
        "R4_timestamp_sequences",
        "drop orders delivered before purchase; null carrier timestamps that are out of sequence "
        "(post-purchase fields, never used as features)",
        len(orders),
        len(kept),
        carrier_before_purchase_nulled=int(before_purchase.sum()),
        carrier_after_delivery_nulled=int(after_delivery.sum()),
    )
    return kept


def r5_items(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    products: pd.DataFrame,
    translation: pd.DataFrame,
    log: CleaningLog,
) -> pd.DataFrame:
    items = items[items["order_id"].isin(orders["order_id"])]
    prod = products.copy()
    prod["volume_cm3"] = (
        prod["product_length_cm"] * prod["product_height_cm"] * prod["product_width_cm"]
    )
    english = dict(
        zip(
            translation["product_category_name"],
            translation["product_category_name_english"],
            strict=True,
        )
    )
    category = prod["product_category_name"]
    prod["category"] = category.map(english).fillna(category).fillna(UNKNOWN_CATEGORY)
    merged = items.merge(
        prod[
            [
                "product_id",
                "product_weight_g",
                "volume_cm3",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
                "category",
            ]
        ],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    merged["dims_missing"] = merged["product_weight_g"].isna() | merged["volume_cm3"].isna()
    grouped = merged.groupby("order_id", sort=True)
    agg = grouped.agg(
        n_items=("order_item_id", "size"),
        n_distinct_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("volume_cm3", "sum"),
        max_length_cm=("product_length_cm", "max"),
        max_height_cm=("product_height_cm", "max"),
        max_width_cm=("product_width_cm", "max"),
        dims_missing=("dims_missing", "max"),
    )
    # Any item with unknown dimensions makes the order totals unknown (not silently smaller).
    missing_dims = agg.pop("dims_missing").astype(bool)
    dims = ["total_weight_g", "total_volume_cm3", "max_length_cm", "max_height_cm", "max_width_cm"]
    agg.loc[missing_dims, dims] = np.nan
    primary = stable_sort(merged, ["order_id", "order_item_id"]).drop_duplicates("order_id")
    primary = primary.set_index("order_id")[["product_id", "seller_id", "category"]].rename(
        columns={
            "product_id": "primary_product_id",
            "seller_id": "primary_seller_id",
            "category": "primary_category",
        }
    )
    agg = agg.join(primary)
    kept = orders.merge(
        agg, left_on="order_id", right_index=True, how="inner", validate="one_to_one"
    )
    unknown_category = int((kept["primary_category"] == UNKNOWN_CATEGORY).sum())
    untranslated = sorted(
        set(kept["primary_category"]) - set(english.values()) - {UNKNOWN_CATEGORY}
    )
    log.add(
        "R5_items",
        "aggregate items to one row per order (counts, price, freight, weight, volume, primary "
        "item); drop orders without items",
        len(orders),
        len(kept),
        orders_with_unknown_category=unknown_category,
        orders_with_unknown_dimensions=int(missing_dims.reindex(kept["order_id"]).sum()),
        untranslated_categories_kept=untranslated,
    )
    return kept


def r6_payments(orders: pd.DataFrame, payments: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    payments = payments[payments["order_id"].isin(orders["order_id"])]
    invalid = (payments["payment_type"] == "not_defined") | (payments["payment_value"] <= 0)
    valid = payments[~invalid]
    ordered = stable_sort(
        valid, ["order_id", "payment_value", "payment_sequential"], ascending=[True, False, True]
    )
    primary_type = ordered.drop_duplicates("order_id").set_index("order_id")["payment_type"]
    agg = valid.groupby("order_id", sort=True).agg(
        payment_total=("payment_value", "sum"),
        n_payments=("payment_sequential", "size"),
        max_installments=("payment_installments", "max"),
    )
    agg["payment_type"] = primary_type
    kept = orders.merge(
        agg, left_on="order_id", right_index=True, how="inner", validate="one_to_one"
    )
    log.add(
        "R6_payments",
        "drop payment rows typed 'not_defined' or with value <= 0, aggregate per order (type of the "
        "largest payment); drop orders left without a valid payment",
        len(orders),
        len(kept),
        invalid_payment_rows_dropped=int(invalid.sum()),
    )
    return kept


def r7_reviews(orders: pd.DataFrame, reviews: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    reviews = reviews[reviews["order_id"].isin(orders["order_id"])]
    multi = int((reviews.groupby("order_id").size() > 1).sum())
    latest = stable_sort(
        reviews, ["order_id", "review_answer_timestamp", "review_creation_date", "review_id"]
    ).drop_duplicates("order_id", keep="last")
    latest = latest.set_index("order_id")[["review_score"]]
    kept = orders.merge(
        latest, left_on="order_id", right_index=True, how="left", validate="one_to_one"
    )
    log.add(
        "R7_reviews",
        "keep the latest review per order (score only; free text is never loaded into clean data)",
        len(orders),
        len(kept),
        orders_with_multiple_reviews=multi,
        orders_without_review=int(kept["review_score"].isna().sum()),
    )
    return kept


def r8_geography(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    sellers: pd.DataFrame,
    centroids: pd.DataFrame,
    log: CleaningLog,
) -> pd.DataFrame:
    cust = customers[
        ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_state"]
    ].rename(columns={"customer_zip_code_prefix": "customer_zip_prefix"})
    sell = sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]].rename(
        columns={"seller_id": "primary_seller_id", "seller_zip_code_prefix": "seller_zip_prefix"}
    )
    kept = orders.merge(cust, on="customer_id", how="inner", validate="one_to_one")
    kept = kept.merge(sell, on="primary_seller_id", how="left", validate="many_to_one")
    c = centroids.reindex(kept["customer_zip_prefix"])
    s = centroids.reindex(kept["seller_zip_prefix"])
    kept["distance_km"] = haversine_km(c["lat"], c["lng"], s["lat"], s["lng"])
    kept["same_state"] = kept["customer_state"] == kept["seller_state"]
    log.add(
        "R8_geography",
        "join customer and primary-seller geography; distance between zip-prefix centroids "
        "(null when a prefix is unknown); city names dropped",
        len(orders),
        len(kept),
        distance_unknown=int(kept["distance_km"].isna().sum()),
    )
    return kept


def r9_derive(orders: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = orders.copy()
    purchase = df["order_purchase_timestamp"]
    df["purchase_month"] = purchase.dt.strftime("%Y-%m")
    df["purchase_dow"] = purchase.dt.dayofweek.astype("int64")
    df["purchase_hour"] = purchase.dt.hour.astype("int64")
    day = purchase.dt.normalize()
    df["promised_days"] = (df["order_estimated_delivery_date"].dt.normalize() - day).dt.days
    df["delivery_days"] = (df["order_delivered_customer_date"].dt.normalize() - day).dt.days
    df["is_late"] = df["delivery_days"] > df["promised_days"]
    df["bad_review"] = (df["review_score"] <= 2).astype("boolean")
    df.loc[df["review_score"].isna(), "bad_review"] = pd.NA
    log.add(
        "R9_derive",
        "derive purchase time parts, promised_days and delivery_days (calendar days) and is_late "
        "= delivery_days > promised_days",
        len(orders),
        len(df),
        late_orders=int(df["is_late"].sum()),
        late_rate=round(float(df["is_late"].mean()), 6),
    )
    return df
