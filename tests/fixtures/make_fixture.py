"""Build tests/fixtures/olist_mini: a deterministic real subset of Olist with injected defects.

Selection: for every purchase month, the first PER_MONTH order ids ordered by sha1(order_id), plus all
rows that reference them (items, payments, reviews, customers, products, sellers, <= 3 geolocation
rows per referenced zip prefix). Raw strings are copied unchanged, then a short list of known
defects is injected and recorded in injections.json so tests can assert how each one is handled.

Usage (needs the real data in data/raw): uv run python tests/fixtures/make_fixture.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from olist_core.cleaning.geo import build_centroid_lookup  # noqa: E402
from olist_core.download import RAW_FILES, download_raw  # noqa: E402

PER_MONTH = 120
SRC = ROOT / "data" / "raw"
OUT = Path(__file__).resolve().parent / "olist_mini"


def read_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(SRC / name, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def pick(df: pd.DataFrame, n: int, offset: int = 0) -> str:
    """Deterministic pick of an order id from a candidate frame."""
    ids = sorted(df["order_id"], key=sha1)
    return ids[offset % len(ids)] if n else ids[0]


def main() -> None:
    orders = read_raw("olist_orders_dataset.csv")
    orders["_month"] = orders["order_purchase_timestamp"].str[:7]
    orders["_key"] = orders["order_id"].map(sha1)
    orders = orders.sort_values(["_month", "_key"], kind="mergesort")
    sample = orders.groupby("_month", sort=True).head(PER_MONTH).drop(columns=["_month", "_key"])
    ids = set(sample["order_id"])

    items = read_raw("olist_order_items_dataset.csv")
    items = items[items["order_id"].isin(ids)]
    payments = read_raw("olist_order_payments_dataset.csv")
    payments = payments[payments["order_id"].isin(ids)]
    reviews = read_raw("olist_order_reviews_dataset.csv")
    reviews = reviews[reviews["order_id"].isin(ids)]
    customers = read_raw("olist_customers_dataset.csv")
    customers = customers[customers["customer_id"].isin(sample["customer_id"])]
    products = read_raw("olist_products_dataset.csv")
    products = products[products["product_id"].isin(items["product_id"])]
    sellers = read_raw("olist_sellers_dataset.csv")
    sellers = sellers[sellers["seller_id"].isin(items["seller_id"])]
    # Raw zip prefixes are not consistently zero-padded across tables: compare padded values.
    prefixes = {
        p.zfill(5)
        for p in set(customers["customer_zip_code_prefix"]) | set(sellers["seller_zip_code_prefix"])
    }
    geo = read_raw("olist_geolocation_dataset.csv")
    geo = geo[geo["geolocation_zip_code_prefix"].str.zfill(5).isin(prefixes)]
    geo = geo.groupby("geolocation_zip_code_prefix", sort=False).head(3)
    translation = read_raw("product_category_name_translation.csv")

    # --- injections (targets: delivered orders purchased in 2017-06..2017-12, all distinct) -----
    mid = sample[
        (sample["order_status"] == "delivered")
        & (sample["order_purchase_timestamp"].str[:7].between("2017-06", "2017-12"))
        & (sample["order_delivered_customer_date"] != "")
        & (sample["order_delivered_carrier_date"] != "")
    ]
    single_payment = payments.groupby("order_id").size()
    single_payment = set(single_payment[single_payment == 1].index)
    reviewed = set(reviews["order_id"])
    candidates = sorted(mid["order_id"], key=sha1)
    candidates = [o for o in candidates if o in single_payment and o in reviewed]
    targets = dict(
        zip(
            [
                "delivered_before_purchase",
                "missing_delivery_date",
                "carrier_before_purchase",
                "no_items",
                "only_invalid_payment",
                "zero_payment_row",
                "duplicate_review",
                "null_category",
                "untranslated_category",
            ],
            candidates[:9],
            strict=True,
        )
    )

    def set_order(order_id: str, column: str, value: str) -> None:
        sample.loc[sample["order_id"] == order_id, column] = value

    purchase = sample.set_index("order_id")["order_purchase_timestamp"]
    t = pd.Timestamp(purchase[targets["delivered_before_purchase"]]) - pd.Timedelta(days=1)
    set_order(targets["delivered_before_purchase"], "order_delivered_customer_date", str(t))
    set_order(targets["missing_delivery_date"], "order_delivered_customer_date", "")
    t = pd.Timestamp(purchase[targets["carrier_before_purchase"]]) - pd.Timedelta(hours=2)
    set_order(targets["carrier_before_purchase"], "order_delivered_carrier_date", str(t))
    items = items[items["order_id"] != targets["no_items"]]
    payments.loc[payments["order_id"] == targets["only_invalid_payment"], "payment_type"] = (
        "not_defined"
    )
    row = payments[payments["order_id"] == targets["zero_payment_row"]].iloc[0].copy()
    row["payment_sequential"] = str(int(row["payment_sequential"]) + 1)
    row["payment_value"] = "0.00"
    payments = pd.concat([payments, row.to_frame().T], ignore_index=True)
    review = reviews[reviews["order_id"] == targets["duplicate_review"]].iloc[0].copy()
    review["review_id"] = sha1("injected-review")[:32]
    review["review_score"] = "1"
    answer = pd.Timestamp(review["review_answer_timestamp"]) + pd.Timedelta(days=3)
    review["review_answer_timestamp"] = str(answer)
    reviews = pd.concat([reviews, review.to_frame().T], ignore_index=True)

    def primary_product(order_id: str) -> str:
        rows = items[items["order_id"] == order_id]
        return rows.sort_values("order_item_id", key=lambda s: s.astype(int)).iloc[0]["product_id"]

    null_cat_product = primary_product(targets["null_category"])
    products.loc[products["product_id"] == null_cat_product, "product_category_name"] = ""
    gamer_product = primary_product(targets["untranslated_category"])
    products.loc[products["product_id"] == gamer_product, "product_category_name"] = "pc_gamer"
    first_geo = geo.iloc[0].copy()
    out_of_brazil = first_geo.copy()
    out_of_brazil["geolocation_lat"], out_of_brazil["geolocation_lng"] = "40.4168", "-3.7038"
    geo = pd.concat([geo, first_geo.to_frame().T, out_of_brazil.to_frame().T], ignore_index=True)

    # --- write -----------------------------------------------------------------------------------
    raw_out = OUT / "raw"
    if OUT.exists():
        shutil.rmtree(OUT)
    raw_out.mkdir(parents=True)
    frames = {
        "olist_orders_dataset.csv": sample,
        "olist_order_items_dataset.csv": items,
        "olist_order_payments_dataset.csv": payments,
        "olist_order_reviews_dataset.csv": reviews,
        "olist_customers_dataset.csv": customers,
        "olist_products_dataset.csv": products,
        "olist_sellers_dataset.csv": sellers,
        "olist_geolocation_dataset.csv": geo,
        "product_category_name_translation.csv": translation,
    }
    assert set(frames) == set(RAW_FILES)
    for name, frame in frames.items():
        frame.to_csv(raw_out / name, index=False, lineterminator="\n", encoding="utf-8")
    shutil.copyfile(SRC / "raw_schema.json", raw_out / "raw_schema.json")
    download_raw(raw_out, update_manifest=True, log=lambda *_: None)
    lookup = build_centroid_lookup(raw_out, OUT / "lookups")
    injections = {
        "orders": targets,
        "null_category_product": null_cat_product,
        "untranslated_category_product": gamer_product,
        "geolocation": {
            "duplicate_rows_added": 1,
            "out_of_brazil_rows_added": 1,
            "prefix": first_geo["geolocation_zip_code_prefix"],
        },
        "per_month": PER_MONTH,
        "orders_total": int(len(sample)),
    }
    (OUT / "injections.json").write_text(
        json.dumps(injections, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"fixture: {len(sample)} orders, {len(items)} items, lookup {lookup['rows']} prefixes")


if __name__ == "__main__":
    main()
