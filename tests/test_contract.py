import json
import shutil

import pandas as pd
import pytest

from olist_core.columns import read_csv_strings
from olist_core.io import read_json
from olist_core.policy import FORBIDDEN_FEATURES, REQUIRED_FEATURES
from olist_core.validators.contract_gate import validate_clean_data_against_contract


def test_contract_round_trip_passes(analyst_out):
    report = validate_clean_data_against_contract(
        analyst_out / "clean_data.csv", analyst_out / "dataset_contract.json"
    )
    assert report.passed, report.errors


def test_contract_content(analyst_out):
    contract = read_json(analyst_out / "dataset_contract.json")
    text = (analyst_out / "dataset_contract.json").read_text(encoding="utf-8")
    assert "run_id" not in text and "generated_at" not in text
    assert contract["primary_key"] == ["order_id"]
    assert [f["name"] for f in contract["required_features"]] == [
        f["name"] for f in REQUIRED_FEATURES
    ]
    assert contract["forbidden_features"] == list(FORBIDDEN_FEATURES)
    for block in ("assumptions", "business_rules", "column_notes", "known_limitations"):
        assert contract[block]["items"], block
        assert contract[block]["narrative_source"] == "template"
    states = next(c for c in contract["columns"] if c["name"] == "customer_state")
    assert len(states["allowed_values"]) == 27
    assert contract["target"]["split_policy"]["test"] == ["2018-05-01", "2018-07-31"]


def _mutate(frame: pd.DataFrame, case: str) -> pd.DataFrame:
    df = frame.copy()
    if case == "drop_column":
        return df.drop(columns=["promised_days"])
    if case == "extra_column":
        df["surprise"] = "x"
    elif case == "bad_dtype":
        df.loc[0, "n_items"] = "abc"
    elif case == "null":
        df.loc[0, "customer_state"] = ""
    elif case == "bounds":
        df.loc[0, "review_score"] = "9"
    elif case == "enum":
        df.loc[0, "payment_type"] = "pix"
    elif case == "pattern":
        df.loc[0, "customer_zip_prefix"] = "12-34"
    elif case == "row_count":
        return df.iloc[1:]
    elif case == "target":
        df.loc[0, "is_late"] = "True" if df.loc[0, "is_late"] == "False" else "False"
    elif case == "duplicate_key":
        df.loc[1, "order_id"] = df.loc[0, "order_id"]
    return df


@pytest.mark.parametrize(
    ("case", "failed_check"),
    [
        ("drop_column", "columns_match"),
        ("extra_column", "columns_match"),
        ("bad_dtype", "dtypes"),
        ("null", "non_nullable"),
        ("bounds", "bounds"),
        ("enum", "allowed_values"),
        ("pattern", "patterns"),
        ("row_count", "row_count"),
        ("target", "constraint_C06"),
        ("duplicate_key", "constraint_C01"),
    ],
)
def test_gate_detects_each_failure_mode(analyst_out, tmp_path, case, failed_check):
    frame = read_csv_strings(analyst_out / "clean_data.csv")
    _mutate(frame, case).to_csv(tmp_path / "clean_data.csv", index=False, lineterminator="\n")
    shutil.copyfile(analyst_out / "dataset_contract.json", tmp_path / "dataset_contract.json")
    report = validate_clean_data_against_contract(
        tmp_path / "clean_data.csv", tmp_path / "dataset_contract.json"
    )
    failed = {c["name"] for c in report.checks if not c["passed"]}
    assert not report.passed
    assert failed_check in failed, report.errors
    assert all(e.startswith("gate_contract: ") for e in report.errors)


def test_gate_never_raises_on_unreadable_inputs(tmp_path, analyst_out):
    (tmp_path / "dataset_contract.json").write_text("{not json", encoding="utf-8")
    report = validate_clean_data_against_contract(
        analyst_out / "clean_data.csv", tmp_path / "dataset_contract.json"
    )
    assert not report.passed and report.checks[0]["name"] == "contract_readable"
    shutil.copyfile(analyst_out / "dataset_contract.json", tmp_path / "ok.json")
    report = validate_clean_data_against_contract(tmp_path / "missing.csv", tmp_path / "ok.json")
    assert not report.passed and report.checks[-1]["name"] == "data_readable"
    json.loads((tmp_path / "ok.json").read_text(encoding="utf-8"))
