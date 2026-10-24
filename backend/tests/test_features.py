import copy

import pandas as pd
import pytest

from pulse_core.columns import read_clean_data
from pulse_core.features.build import LeakageError, check_no_leakage
from pulse_core.io import read_json, write_json_lf
from pulse_core.splits import assign_split
from pulse_core.validators.features_gate import validate_features


def test_leakage_check_rejects_post_decision_sources(analyst_out, spec):
    contract = read_json(analyst_out / "dataset_contract.json")
    check_no_leakage(contract, spec.feature_definitions)
    tampered = copy.deepcopy(contract)
    for column in tampered["columns"]:
        if column["name"] == "x_num":
            column["available_at"] = "post_decision"
    with pytest.raises(LeakageError):
        check_no_leakage(tampered, spec.feature_definitions)


def test_post_decision_columns_do_not_change_features(analyst_out, spec):
    clean = read_clean_data(analyst_out / "clean_data.csv", spec.dtypes)
    before = spec.derive_features(clean)
    future = clean.copy()
    future["post_val"] = future["post_val"] + 1.0
    future["y"] = ~future["y"].astype(bool)
    pd.testing.assert_frame_equal(before, spec.derive_features(future))


def test_split_boundaries(spec):
    ts = pd.to_datetime(
        [
            "2019-12-31 23:59:59",
            "2020-01-01 00:00:00",
            "2020-08-31 23:59:59",
            "2020-09-01 00:00:00",
            "2020-11-30 23:00:00",
            "2020-12-01 00:00:00",
            "2021-01-01 00:00:00",
        ]
    )
    assert assign_split(pd.Series(ts), spec.splits).tolist() == [
        "excluded",
        "train",
        "train",
        "test",
        "test",
        "late_buffer",
        "excluded",
    ]


def test_features_csv_layout(pipeline_out, spec):
    df = pd.read_csv(pipeline_out / "features.csv", nrows=5)
    assert list(df.columns) == [
        spec.id_column,
        spec.time_column,
        spec.split_column,
        *spec.model_features,
        spec.target,
    ]
    feature_spec = read_json(pipeline_out / "feature_spec.json")
    assert feature_spec["input_validation"]["passed"] is True
    assert feature_spec["narrative"]["narrative_source"] == "template"
    assert feature_spec["feature_available_at"] == spec.decision_stage
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
    features = pd.read_csv(pipeline_copy / "features.csv", dtype=str, keep_default_na=False)
    feature_spec = read_json(pipeline_copy / "feature_spec.json")
    if case == "missing_feature":
        features = features.drop(columns=["x_num"])
    elif case == "leaky_feature":
        clean = pd.read_csv(pipeline_copy / "clean_data.csv", dtype=str, usecols=["post_val"])
        features["post_val"] = clean["post_val"]
        feature_spec["model_features"].append("post_val")
        feature_spec["features"].append(
            {"name": "post_val", "kind": "numeric", "derived_from": ["post_val"]}
        )
    elif case == "wrong_split":
        features.loc[0, "split"] = "test"
    elif case == "contract_changed":
        feature_spec["contract_sha256"] = "0" * 64
    elif case == "null_feature":
        features.loc[0, "x_num"] = ""
    features.to_csv(pipeline_copy / "features.csv", index=False, lineterminator="\n")
    write_json_lf(pipeline_copy / "feature_spec.json", feature_spec)
    report = _gate(pipeline_copy)
    failed = {c["name"] for c in report.checks if not c["passed"]}
    assert not report.passed and failed_check in failed, report.errors
    assert all(e.startswith("gate_required_features: ") for e in report.errors)
