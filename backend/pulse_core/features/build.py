"""Build features.csv and feature_spec.json from the validated clean data and the contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pulse_core.columns import COLUMNS_BY_NAME, read_clean_data
from pulse_core.features.derive import derive_checkout_features
from pulse_core.features.spec import (
    FEATURE_DEFINITIONS,
    ID_COLUMNS,
    LEAKAGE_RULE,
    MODEL_FEATURES,
    NUMERIC,
)
from pulse_core.io import read_json, sha256_file, write_csv_stable, write_json_lf
from pulse_core.policy import SPLIT_POLICY, TARGET, assign_split
from pulse_core.validators.contract_gate import validate_clean_data_against_contract

SPEC_VERSION = "1.0"


class LeakageError(ValueError):
    """A feature would be derived from a column that is not available at checkout."""


def check_no_leakage(contract: dict) -> None:
    available = {c["name"]: c["available_at"] for c in contract["columns"]}
    forbidden = set(contract["forbidden_features"])
    for feature in FEATURE_DEFINITIONS:
        if feature["name"] in forbidden:
            raise LeakageError(f"feature {feature['name']} is forbidden by the contract")
        for source in feature["derived_from"]:
            if available.get(source) != "checkout":
                raise LeakageError(
                    f"feature {feature['name']} uses {source} (available_at="
                    f"{available.get(source)!r}, expected 'checkout')"
                )


def _feature_stats(features: pd.DataFrame) -> list[dict]:
    stats = []
    for definition in FEATURE_DEFINITIONS:
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


def build_features(clean_csv: str | Path, contract_json: str | Path, out_dir: str | Path) -> dict:
    """Validate the handoff, derive features, write features.csv and feature_spec.json."""
    out_dir = Path(out_dir)
    input_validation = validate_clean_data_against_contract(clean_csv, contract_json)
    contract = read_json(contract_json)
    check_no_leakage(contract)

    clean = read_clean_data(clean_csv)
    features = derive_checkout_features(clean)
    table = pd.concat(
        [
            clean[["order_id", "purchase_ts"]],
            assign_split(clean["purchase_ts"]).rename("split"),
            features,
            clean[TARGET].astype(bool).astype("int64").rename(TARGET),
        ],
        axis=1,
    )
    write_csv_stable(table, out_dir / "features.csv")

    split = table["split"]
    positive_rate = {
        name: float(table.loc[split == name, TARGET].mean())
        for name in SPLIT_POLICY
        if name in ("train", "test", "censored_buffer")
    }
    counts = {name: int((split == name).sum()) for name in ("train", "test", "censored_buffer")}
    excluded = [
        {"name": c.name, "reason": f"available_at={c.available_at}"}
        for c in COLUMNS_BY_NAME.values()
        if c.available_at != "checkout"
    ]
    spec = {
        "spec_version": SPEC_VERSION,
        "grain": "one row per delivered order (same rows as clean_data.csv)",
        "id_columns": ID_COLUMNS,
        "time_column": "purchase_ts",
        "target": {"name": TARGET, "positive_rate": positive_rate},
        "split": {"policy": SPLIT_POLICY, "rows": counts},
        "model_features": MODEL_FEATURES,
        "features": _feature_stats(features),
        "excluded_columns": excluded,
        "leakage_rule": LEAKAGE_RULE,
        "contract_version": contract["contract_version"],
        "contract_sha256": sha256_file(contract_json),
        "input_validation": input_validation.to_dict(),
    }
    write_json_lf(out_dir / "feature_spec.json", spec)
    return {
        "rows": int(len(table)),
        "features": len(MODEL_FEATURES),
        "input_validation": input_validation.summary,
        "split_rows": counts,
        "positive_rate": positive_rate,
    }
