"""Build features.csv and feature_spec.json from the validated clean data and the contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pulse_core.columns import read_clean_data
from pulse_core.io import read_json, sha256_file, write_csv_stable, write_json_lf
from pulse_core.spec import DatasetSpec
from pulse_core.splits import assign_split, split_policy
from pulse_core.validators.contract_gate import validate_clean_data_against_contract
from pulse_core.validators.report import LeakageError

SPEC_VERSION = "1.0"
NUMERIC = "numeric"
CATEGORICAL = "categorical"

__all__ = ["LeakageError", "build_features", "check_no_leakage"]


def check_no_leakage(contract: dict, feature_definitions) -> None:
    """Every feature derives only from columns tagged with the contract's decision stage."""
    stage = contract["feature_rules"]["feature_available_at"]
    available = {c["name"]: c["available_at"] for c in contract["columns"]}
    forbidden = set(contract["forbidden_features"])
    for feature in feature_definitions:
        if feature["name"] in forbidden:
            raise LeakageError(f"feature {feature['name']} is forbidden by the contract")
        for source in feature["derived_from"]:
            if available.get(source) != stage:
                raise LeakageError(
                    f"feature {feature['name']} uses {source} (available_at="
                    f"{available.get(source)!r}, expected {stage!r})"
                )


def _feature_stats(features: pd.DataFrame, definitions) -> list[dict]:
    stats = []
    for definition in definitions:
        name = definition["name"]
        column = features[name]
        entry = {**definition, "null_rate": float(column.isna().mean())}
        if definition["kind"] == NUMERIC:
            entry.update(
                min=float(column.min()), max=float(column.max()), median=float(column.median())
            )
        else:
            entry["categories"] = sorted(column.dropna().unique().tolist())
        stats.append(entry)
    return stats


def build_features(
    spec: DatasetSpec, clean_csv: str | Path, contract_json: str | Path, out_dir: str | Path
) -> dict:
    """Validate the handoff, derive features, write features.csv and feature_spec.json."""
    out_dir = Path(out_dir)
    input_validation = validate_clean_data_against_contract(clean_csv, contract_json)
    contract = read_json(contract_json)
    check_no_leakage(contract, spec.feature_definitions)

    clean = read_clean_data(clean_csv, spec.dtypes)
    features = spec.derive_features(clean)[spec.model_features]
    table = pd.concat(
        [
            clean[[spec.id_column, spec.time_column]],
            assign_split(clean[spec.time_column], spec.splits).rename(spec.split_column),
            features,
            clean[spec.target].astype(bool).astype("int64").rename(spec.target),
        ],
        axis=1,
    )
    write_csv_stable(table, out_dir / "features.csv")

    split = table[spec.split_column]
    positive_rate = {
        name: float(table.loc[split == name, spec.target].mean()) if (split == name).any() else None
        for name in spec.splits
    }
    counts = {name: int((split == name).sum()) for name in spec.splits}
    excluded = [
        {"name": c.name, "reason": f"available_at={c.available_at}"}
        for c in spec.columns
        if c.available_at != spec.decision_stage
    ]
    feature_spec = {
        "spec_version": SPEC_VERSION,
        "grain": f"{spec.grain} (same rows as clean_data.csv)",
        "id_column": spec.id_column,
        "time_column": spec.time_column,
        "split_column": spec.split_column,
        "id_columns": [spec.id_column, spec.time_column, spec.split_column],
        "feature_available_at": spec.decision_stage,
        "target": {"name": spec.target, "positive_rate": positive_rate},
        "split": {"policy": split_policy(spec), "rows": counts},
        "model_features": spec.model_features,
        "features": _feature_stats(features, spec.feature_definitions),
        "excluded_columns": excluded,
        "leakage_rule": contract["feature_rules"]["leakage_rule"],
        "contract_version": contract["contract_version"],
        "contract_sha256": sha256_file(contract_json),
        "input_validation": input_validation.to_dict(),
    }
    write_json_lf(out_dir / "feature_spec.json", feature_spec)
    return {
        "rows": int(len(table)),
        "features": len(spec.model_features),
        "input_validation": input_validation.summary,
        "split_rows": counts,
        "positive_rate": positive_rate,
    }
