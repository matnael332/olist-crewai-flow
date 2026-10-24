"""Deliberate fault injection, to demonstrate (and test) that every gate fails gracefully.

Each fault corrupts one thing after the step that produced it, so the next gate must catch it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from pulse_core.io import read_json, write_json_lf

FAULTS = {
    "bad_input_hash": "check_inputs",
    "crew_error": "run_analyst_stage",
    "drop_column": "gate_contract",
    "contract_mismatch": "gate_contract",
    "missing_feature": "gate_required_features",
    "leaky_feature": "gate_required_features",
    "missing_artifact": "gate_artifacts",
}


class InjectedQuotaError(RuntimeError):
    """Simulates Gemini rejecting the call with 429 RESOURCE_EXHAUSTED."""


def tampered_raw_copy(raw_dir: Path, work_dir: Path) -> Path:
    """Copy the raw folder and append one byte to the orders file (hash mismatch)."""
    target = work_dir / "tampered_raw"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(raw_dir, target)
    with open(target / "olist_orders_dataset.csv", "ab") as fh:
        fh.write(b"\n")
    return target


def _rewrite_csv(path: Path, transform) -> None:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    transform(frame).to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def after_analyst(fault: str, art: Path) -> bool:
    if fault == "drop_column":
        _rewrite_csv(art / "clean_data.csv", lambda df: df.drop(columns=["promised_days"]))
        return True
    if fault == "contract_mismatch":
        contract = read_json(art / "dataset_contract.json")
        for column in contract["columns"]:
            if column["name"] == "payment_type":
                column["allowed_values"] = [v for v in column["allowed_values"] if v != "boleto"]
        write_json_lf(art / "dataset_contract.json", contract)
        return True
    return False


def after_features(fault: str, art: Path) -> bool:
    if fault == "missing_feature":
        _rewrite_csv(art / "features.csv", lambda df: df.drop(columns=["promised_days"]))
        return True
    if fault == "leaky_feature":
        delivery = pd.read_csv(art / "clean_data.csv", dtype=str, usecols=["delivery_days"])

        def add(df: pd.DataFrame) -> pd.DataFrame:
            df["delivery_days"] = delivery["delivery_days"].to_numpy()
            return df

        _rewrite_csv(art / "features.csv", add)
        spec = read_json(art / "feature_spec.json")
        spec["model_features"].append("delivery_days")
        spec["features"].append(
            {"name": "delivery_days", "kind": "numeric", "derived_from": ["delivery_days"]}
        )
        write_json_lf(art / "feature_spec.json", spec)
        return True
    return False


def after_modeling(fault: str, art: Path) -> bool:
    if fault == "missing_artifact":
        (art / "model_card.md").unlink(missing_ok=True)
        return True
    return False
