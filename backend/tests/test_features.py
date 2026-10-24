import copy

import pandas as pd
import pytest

from pulse_core.columns import read_clean_data
from pulse_core.features.build import LeakageError, check_no_leakage
from pulse_core.features.derive import derive_checkout_features
from pulse_core.features.spec import MODEL_FEATURES
from pulse_core.io import read_json
from pulse_core.policy import REQUIRED_FEATURES, assign_split
from pulse_core.validators.features_gate import validate_features


def test_catalogue_covers_required_features():
    assert {f["name"] for f in REQUIRED_FEATURES} <= set(MODEL_FEATURES)


def test_leakage_check_rejects_post_purchase_sources(analyst_out):
    contract = read_json(analyst_out / "dataset_contract.json")
    check_no_leakage(contract)
    tampered = copy.deepcopy(contract)
    for column in tampered["columns"]:
        if column["name"] == "promised_days":
            column["available_at"] = "post_purchase"
    with pytest.raises(LeakageError):
        check_no_leakage(tampered)


def test_post_purchase_columns_do_not_change_features(analyst_out):
    clean = read_clean_data(analyst_out / "clean_data.csv")
    before = derive_checkout_features(clean)
    future = clean.copy()
    future["delivered_customer_ts"] = future["delivered_customer_ts"] + pd.Timedelta(days=30)
    future["delivery_days"] = future["delivery_days"] + 30
    future["review_score"] = 1
    future["is_late"] = ~future["is_late"].astype(bool)
    pd.testing.assert_frame_equal(before, derive_checkout_features(future))


def test_split_boundaries():
    ts = pd.to_datetime(
        [
            "2016-12-31 23:59:59",
            "2017-01-01 00:00:00",
            "2018-04-30 23:59:59",
            "2018-05-01 00:00:00",
            "2018-07-31 23:00:00",
            "2018-08-01 00:00:00",
            "2018-09-01 00:00:00",
        ]
    )
    assert assign_split(pd.Series(ts)).tolist() == [
        "excluded",
        "train",
        "train",
        "test",
        "test",
        "censored_buffer",
        "excluded",
    ]


def test_features_csv_layout(pipeline_out):
    df = pd.read_csv(pipeline_out / "features.csv", nrows=5)
    assert list(df.columns) == ["order_id", "purchase_ts", "split", *MODEL_FEATURES, "is_late"]
    spec = read_json(pipeline_out / "feature_spec.json")
    assert spec["input_validation"]["passed"] is True
    assert spec["narrative"]["narrative_source"] == "template"
    assert b"\r" not in (pipeline_out / "features.csv").read_bytes()


def _gate(path):
    return validate_features(
        path / "features.csv", path / "feature_spec.json", path / "dataset_contract.json"
    )


def test_features_gate_passes(pipeline_out):
    report = _gate(pipeline_out)
    assert report.passed, report.errors


@pytest.mark.parametrize(
    ("case", "failed_check"),
    [
        ("missing_feature", "required_features_present"),
        ("leaky_feature", "no_forbidden_model_features"),
        ("wrong_split", "split_matches_policy"),
        ("contract_changed", "contract_hash"),
        ("null_feature", "non_nullable_features"),
    ],
)
def test_features_gate_failure_modes(pipeline_copy, case, failed_check):
    from pulse_core.io import write_json_lf

    features = pd.read_csv(pipeline_copy / "features.csv", dtype=str, keep_default_na=False)
    spec = read_json(pipeline_copy / "feature_spec.json")
    if case == "missing_feature":
        features = features.drop(columns=["promised_days"])
    elif case == "leaky_feature":
        clean = pd.read_csv(pipeline_copy / "clean_data.csv", dtype=str, usecols=["delivery_days"])
        features["delivery_days"] = clean["delivery_days"]
        spec["model_features"].append("delivery_days")
        spec["features"].append(
            {"name": "delivery_days", "kind": "numeric", "derived_from": ["delivery_days"]}
        )
    elif case == "wrong_split":
        features.loc[0, "split"] = "test"
    elif case == "contract_changed":
        spec["contract_sha256"] = "0" * 64
    elif case == "null_feature":
        features.loc[0, "promised_days"] = ""
    features.to_csv(pipeline_copy / "features.csv", index=False, lineterminator="\n")
    write_json_lf(pipeline_copy / "feature_spec.json", spec)
    report = _gate(pipeline_copy)
    failed = {c["name"] for c in report.checks if not c["passed"]}
    assert not report.passed and failed_check in failed, report.errors
    assert all(e.startswith("gate_required_features: ") for e in report.errors)
