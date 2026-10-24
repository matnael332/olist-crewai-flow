"""Deliberate fault injection, to demonstrate (and test) that every gate fails gracefully.

Each fault corrupts one thing after the step that produced it, so the next gate must catch it.
The targets are read from the artifacts (manifest, contract, feature spec), never hard-coded.
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
    """Simulates the LLM provider rejecting the call with 429 RESOURCE_EXHAUSTED."""


def tampered_raw_copy(raw_dir: Path, work_dir: Path) -> Path:
    """Copy the raw folder and append one byte to the first manifest file (hash mismatch)."""
    target = work_dir / "tampered_raw"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(raw_dir, target)
    first = sorted(read_json(target / "manifest.json")["files"])[0]
    with open(target / first, "ab") as fh:
        fh.write(b"\n")
    return target


def _rewrite_csv(path: Path, transform) -> None:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    transform(frame).to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def _first_required_feature(contract: dict) -> str:
    return contract["required_features"][0]["name"]


def after_analyst(fault: str, art: Path) -> bool:
    if fault == "drop_column":
        column = _first_required_feature(read_json(art / "dataset_contract.json"))
        _rewrite_csv(art / "clean_data.csv", lambda df: df.drop(columns=[column]))
        return True
    if fault == "contract_mismatch":
        contract = read_json(art / "dataset_contract.json")
        for column in contract["columns"]:
            if column["dtype"] == "string" and isinstance(column.get("allowed_values"), list):
                column["allowed_values"] = column["allowed_values"][1:]
                break
        else:
            raise ValueError("contract_mismatch needs a string column with allowed values")
        write_json_lf(art / "dataset_contract.json", contract)
        return True
    return False


def after_features(fault: str, art: Path) -> bool:
    if fault == "missing_feature":
        column = _first_required_feature(read_json(art / "dataset_contract.json"))
        _rewrite_csv(art / "features.csv", lambda df: df.drop(columns=[column]))
        return True
    if fault == "leaky_feature":
        contract = read_json(art / "dataset_contract.json")
        clean_columns = pd.read_csv(art / "clean_data.csv", nrows=0).columns
        target = contract["target"]["name"]
        leak = next(c for c in contract["forbidden_features"] if c != target and c in clean_columns)
        values = pd.read_csv(art / "clean_data.csv", dtype=str, usecols=[leak])

        def add(df: pd.DataFrame) -> pd.DataFrame:
            df[leak] = values[leak].to_numpy()
            return df

        _rewrite_csv(art / "features.csv", add)
        spec = read_json(art / "feature_spec.json")
        spec["model_features"].append(leak)
        spec["features"].append({"name": leak, "kind": "numeric", "derived_from": [leak]})
        write_json_lf(art / "feature_spec.json", spec)
        return True
    return False


def after_modeling(fault: str, art: Path) -> bool:
    if fault == "missing_artifact":
        (art / "model_card.md").unlink(missing_ok=True)
        return True
    return False
