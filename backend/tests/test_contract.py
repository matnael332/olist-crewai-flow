import json
import shutil

import pandas as pd
import pytest

from pulse_core.columns import read_csv_strings
from pulse_core.io import read_json
from pulse_core.validators.contract_gate import validate_clean_data_against_contract


def test_contract_round_trip_passes(analyst_out):
    report = validate_clean_data_against_contract(
        analyst_out / "clean_data.csv", analyst_out / "dataset_contract.json"
    )
    assert report.passed, report.errors


def test_contract_content(analyst_out, spec):
    contract = read_json(analyst_out / "dataset_contract.json")
    text = (analyst_out / "dataset_contract.json").read_text(encoding="utf-8")
    assert "run_id" not in text and "generated_at" not in text
    assert contract["primary_key"] == [spec.id_column]
    assert [f["name"] for f in contract["required_features"]] == [
        f["name"] for f in spec.required_features
    ]
    assert contract["forbidden_features"] == list(spec.forbidden_features)
    for block in ("assumptions", "business_rules", "column_notes", "known_limitations"):
        assert contract[block]["items"], block
        assert contract[block]["narrative_source"] == "template"
    cat = next(c for c in contract["columns"] if c["name"] == "x_cat")
    assert cat["allowed_values"] == ["a", "b", "c"]
    flag = next(c for c in contract["columns"] if c["name"] == "x_flag")
    assert flag["allowed_values"] == [False, True]
    assert contract["target"]["split_policy"]["test"] == list(spec.splits["test"])
    assert contract["feature_rules"]["feature_available_at"] == spec.decision_stage
    assert contract["target"]["entity"] == spec.entity and contract["target"]["label"]


def _mutate(frame: pd.DataFrame, case: str) -> pd.DataFrame:
    df = frame.copy()
    if case == "drop_column":
        return df.drop(columns=["x_num"])
    if case == "extra_column":
        df["surprise"] = "x"
    elif case == "bad_dtype":
        df.loc[0, "x_num"] = "abc"
    elif case == "null":
        df.loc[0, "x_cat"] = ""
    elif case == "bounds":
        df.loc[0, "x_num"] = "99"
    elif case == "enum":
        df.loc[0, "x_cat"] = "zzz"
    elif case == "pattern":
        df.loc[0, "row_id"] = "bad-id"
    elif case == "row_count":
        return df.iloc[1:]
    elif case == "target":
        df.loc[0, "y"] = "True" if df.loc[0, "y"] == "False" else "False"
    elif case == "duplicate_key":
        df.loc[1, "row_id"] = df.loc[0, "row_id"]
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
        ("target", "constraint_C02"),
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
