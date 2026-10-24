"""Gate 2 (before modeling): all required features exist, nothing forbidden is used, and every
feature is derived from checkout-time columns only. Never raises."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pulse_core.io import read_json, sha256_file
from pulse_core.policy import assign_split
from pulse_core.validators.report import GateReport

GATE = "gate_required_features"


def validate_features(
    features_csv: str | Path, spec_json: str | Path, contract_json: str | Path
) -> GateReport:
    report = GateReport(GATE)
    try:
        contract = read_json(contract_json)
        spec = read_json(spec_json)
        report.record("inputs_readable", True, "dataset_contract.json and feature_spec.json parsed")
    except (OSError, ValueError) as exc:
        report.record("inputs_readable", False, f"cannot read contract or feature spec: {exc}")
        return report
    try:
        df = pd.read_csv(features_csv, dtype={"order_id": str, "split": str}, encoding="utf-8")
        report.record("features_readable", True, f"features.csv loaded ({len(df):,} rows)")
    except (OSError, ValueError) as exc:
        report.record("features_readable", False, f"cannot read features.csv: {exc}")
        return report

    columns = list(df.columns)
    model_features = spec.get("model_features", [])
    target = contract["target"]["name"]
    required = [f["name"] for f in contract.get("required_features", [])]
    missing = [name for name in required if name not in columns]
    report.record(
        "required_features_present",
        not missing,
        f"all {len(required)} required features present"
        if not missing
        else f"required feature(s) missing from features.csv: {missing}",
    )
    undeclared = [name for name in required if name not in model_features]
    report.record(
        "required_features_declared",
        not undeclared,
        "required features declared in feature_spec.model_features"
        if not undeclared
        else f"required feature(s) not declared as model features: {undeclared}",
    )
    absent = [name for name in model_features if name not in columns]
    report.record(
        "model_features_present",
        not absent,
        "every declared model feature has a column"
        if not absent
        else f"declared model feature(s) without a column: {absent}",
    )

    forbidden = set(contract.get("forbidden_features", []))
    used = [name for name in model_features if name in forbidden]
    report.record(
        "no_forbidden_model_features",
        not used,
        f"none of the {len(forbidden)} forbidden columns is a model feature"
        if not used
        else f"forbidden feature(s) used by the model: {used}",
    )
    stray = [c for c in columns if c in forbidden and c != target]
    report.record(
        "no_forbidden_columns",
        not stray,
        "features.csv carries no post-purchase columns"
        if not stray
        else f"post-purchase column(s) present in features.csv: {stray}",
    )

    available = {c["name"]: c["available_at"] for c in contract.get("columns", [])}
    leaks = [
        f"{feat['name']} <- {src} ({available.get(src)})"
        for feat in spec.get("features", [])
        for src in feat.get("derived_from", [])
        if available.get(src) != "checkout"
    ]
    report.record(
        "derived_from_checkout_only",
        not leaks,
        "every feature derives from checkout-time columns"
        if not leaks
        else f"features derived from non-checkout columns: {leaks}",
    )

    if target in df:
        values = set(df[target].dropna().unique().tolist())
        report.record(
            "target_binary",
            values <= {0, 1} and df[target].notna().all(),
            f"{target} is binary 0/1" if values <= {0, 1} else f"{target} values {sorted(values)}",
        )
    else:
        report.record("target_binary", False, f"target column {target} missing")

    if {"purchase_ts", "split"} <= set(columns):
        ts = pd.to_datetime(df["purchase_ts"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        mismatched = int((assign_split(ts) != df["split"]).sum())
        report.record(
            "split_matches_policy",
            mismatched == 0,
            "split column matches the contract split policy"
            if mismatched == 0
            else f"{mismatched} rows have a split that differs from the policy",
        )
    else:
        report.record("split_matches_policy", False, "purchase_ts or split column missing")

    nullable = {f["name"]: f.get("nullable", False) for f in spec.get("features", [])}
    null_violations = [
        f"{name} ({int(df[name].isna().sum())})"
        for name in model_features
        if name in df and not nullable.get(name, False) and df[name].isna().any()
    ]
    report.record(
        "non_nullable_features",
        not null_violations,
        "no nulls in non-nullable features"
        if not null_violations
        else f"nulls in non-nullable features: {null_violations}",
    )

    expected = contract.get("row_count")
    report.record(
        "row_count",
        expected == len(df),
        f"{len(df):,} rows match the contract"
        if expected == len(df)
        else f"{len(df):,} rows, contract expects {expected:,}",
    )
    digest = sha256_file(contract_json)
    report.record(
        "contract_hash",
        spec.get("contract_sha256") == digest,
        f"feature spec follows contract sha256 {digest[:12]}"
        if spec.get("contract_sha256") == digest
        else "feature_spec.contract_sha256 does not match dataset_contract.json",
    )
    return report
