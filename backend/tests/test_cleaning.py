import pandas as pd

from olist_core.cleaning.build import run_cleaning
from olist_core.columns import COLUMN_NAMES, read_clean_data
from olist_core.io import read_json
from olist_core.raw import load_raw_tables


def _clean(analyst_out) -> pd.DataFrame:
    return read_clean_data(analyst_out / "clean_data.csv").set_index("order_id", drop=False)


def test_injected_defects_are_handled(analyst_out, injections):
    clean = _clean(analyst_out)
    targets = injections["orders"]
    for dropped in (
        "delivered_before_purchase",
        "missing_delivery_date",
        "no_items",
        "only_invalid_payment",
    ):
        assert targets[dropped] not in clean.index, dropped
    assert targets["zero_payment_row"] in clean.index
    assert clean.loc[targets["duplicate_review"], "review_score"] == 1
    assert pd.isna(clean.loc[targets["carrier_before_purchase"], "delivered_carrier_ts"])
    assert clean.loc[targets["null_category"], "primary_category"] == "unknown"
    assert clean.loc[targets["untranslated_category"], "primary_category"] == "pc_gamer"


def test_rule_counts_match_raw_data(analyst_out, fixture_raw_dir):
    orders = load_raw_tables(fixture_raw_dir, ["orders"])["orders"]
    report = read_json(analyst_out / "cleaning_report.json")
    rules = {r["rule"]: r for r in report["rules"]}
    assert rules["R1_keep_delivered"]["rows_removed"] == int(
        (orders.order_status != "delivered").sum()
    )
    assert rules["R2_require_delivery_date"]["rows_removed"] == 1
    assert rules["R4_timestamp_sequences"]["rows_removed"] == 1
    assert rules["R4_timestamp_sequences"]["detail"]["carrier_before_purchase_nulled"] >= 1
    assert rules["R6_payments"]["detail"]["invalid_payment_rows_dropped"] >= 2
    assert report["rows_out"] == rules["R10_finalize"]["rows_after"]


def test_grain_target_and_columns(analyst_out):
    clean = read_clean_data(analyst_out / "clean_data.csv")
    assert list(clean.columns) == COLUMN_NAMES
    assert clean["order_id"].is_unique
    late = clean["delivery_days"] > clean["promised_days"]
    assert (clean["is_late"].astype(bool) == late.astype(bool)).all()
    assert 0.0 < clean["is_late"].astype(bool).mean() < 0.2
    assert clean["customer_zip_prefix"].str.fullmatch(r"\d{5}").all()


def test_clean_csv_is_byte_identical_and_lf(tmp_path, fixture_raw_dir, fixture_lookups_dir):
    tables = load_raw_tables(fixture_raw_dir)
    run_cleaning(tables, fixture_lookups_dir, tmp_path / "a")
    run_cleaning(tables, fixture_lookups_dir, tmp_path / "b")
    first = (tmp_path / "a" / "clean_data.csv").read_bytes()
    assert first == (tmp_path / "b" / "clean_data.csv").read_bytes()
    assert b"\r" not in first
