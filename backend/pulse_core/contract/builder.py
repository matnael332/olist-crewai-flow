"""Build dataset_contract.json from the written clean_data.csv (round trip).

The contract is derived from the file exactly as the gate will read it back, so observed values can
never disagree with what the gate sees. Binding rules (bounds, enums, patterns, constraints) come
from the dataset spec and are declared, not observed; observed ranges are informational only. No run
id or timestamp is stored, so the contract is byte-identical across reproducible runs.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from pulse_core.columns import DATETIME_FORMAT, read_clean_data
from pulse_core.narratives import ContractNarrative
from pulse_core.spec import DatasetSpec
from pulse_core.splits import assign_split, split_policy


def _observed(series: pd.Series, dtype: str) -> dict | None:
    values = series.dropna()
    if values.empty or dtype not in ("int", "float", "datetime"):
        return None
    lo, hi = values.min(), values.max()
    if dtype == "datetime":
        return {"min": lo.strftime(DATETIME_FORMAT), "max": hi.strftime(DATETIME_FORMAT)}
    if dtype == "float":  # widened outward at 4 decimals: informational, never binding
        return {"min": math.floor(float(lo) * 1e4) / 1e4, "max": math.ceil(float(hi) * 1e4) / 1e4}
    return {"min": int(lo), "max": int(hi)}


def _allowed_values(spec: DatasetSpec, name: str, dtype: str, series: pd.Series) -> list | None:
    declared = spec.allowed_values.get(name)
    if declared == "observed":
        return sorted(series.dropna().unique().tolist())
    if declared is not None:
        return list(declared)
    if dtype == "bool":
        return [False, True]
    return None


def merge_narrative(
    base: ContractNarrative, extra: ContractNarrative | None, known: set[str]
) -> dict:
    """Template items always stay; LLM items are appended when new (case-insensitive).

    ``known`` is the set of clean column names: notes on unknown columns are dropped.
    """

    def merge(a: list[str], b: list[str]) -> list[str]:
        seen = {x.strip().lower() for x in a}
        return a + [x for x in b if x.strip() and x.strip().lower() not in seen]

    notes = [n.model_dump() for n in base.column_notes]
    if extra is not None:
        noted = {(n["name"], n["note"].strip().lower()) for n in notes}
        notes += [
            n.model_dump()
            for n in extra.column_notes
            if n.name in known and (n.name, n.note.strip().lower()) not in noted
        ]
    source = "template+llm" if extra is not None else "template"
    return {
        "assumptions": {
            "narrative_source": source,
            "items": merge(base.assumptions, extra.assumptions if extra else []),
        },
        "business_rules": {
            "narrative_source": source,
            "items": merge(base.business_rules, extra.business_rules if extra else []),
        },
        "column_notes": {"narrative_source": source, "items": notes},
        "known_limitations": {
            "narrative_source": source,
            "items": merge(base.known_limitations, extra.known_limitations if extra else []),
        },
    }


def build_contract(
    spec: DatasetSpec,
    clean_csv: str | Path,
    cleaning_report: dict,
    provenance: dict,
    narrative: dict,
) -> dict:
    clean = read_clean_data(clean_csv, spec.dtypes)
    columns = []
    for col in spec.columns:
        series = clean[col.name]
        entry = {
            "name": col.name,
            "dtype": col.dtype,
            "nullable": col.nullable,
            "available_at": col.available_at,
            "description": col.description,
        }
        if col.name == spec.id_column:
            entry["unique"] = True
        if col.name in spec.patterns:
            entry["pattern"] = spec.patterns[col.name]
        if col.name in spec.bounds:
            entry["bounds"] = dict(spec.bounds[col.name])
        allowed = _allowed_values(spec, col.name, col.dtype, series)
        if allowed is not None:
            entry["allowed_values"] = allowed
        observed = _observed(series, col.dtype)
        if observed is not None:
            entry["observed_range"] = observed
        columns.append(entry)

    split = assign_split(clean[spec.time_column], spec.splits)
    positive = clean[spec.target].astype(bool)
    rates = {"overall": float(positive.mean())}
    for name in spec.splits:
        rates[name] = float(positive[split == name].mean()) if (split == name).any() else None

    time = clean[spec.time_column]
    return {
        "contract_version": spec.contract_version,
        "dataset": {
            "name": spec.name,
            "source": spec.source,
            "license": spec.license,
            "grain": spec.grain,
            "snapshot_date": spec.snapshot_date,
        },
        "provenance": {
            **provenance,
            "cleaning_rules": [
                {"rule": r["rule"], "rows_removed": r["rows_removed"]}
                for r in cleaning_report["rules"]
            ],
        },
        "row_count": int(len(clean)),
        "primary_key": [spec.id_column],
        "date_range": {
            "column": spec.time_column,
            "min": time.min().strftime(DATETIME_FORMAT),
            "max": time.max().strftime(DATETIME_FORMAT),
        },
        "columns": columns,
        "constraints": [dict(c) for c in spec.constraints],
        "population": dict(spec.population),
        **({"data_groups": dict(spec.data_groups)} if spec.data_groups else {}),
        "target": {
            "name": spec.target,
            "label": spec.target_label,
            "task": "binary_classification",
            "entity": spec.entity,
            "decision_time": spec.decision_time,
            "definition": spec.target_definition,
            "rejected_alternative": spec.rejected_target,
            "expected_positive_rate": rates,
            "split_policy": split_policy(spec),
        },
        "required_features": [
            {**feat, "available_at": spec.decision_stage} for feat in spec.required_features
        ],
        "forbidden_features": list(spec.forbidden_features),
        "feature_rules": {
            "feature_available_at": spec.decision_stage,
            "identifiers_excluded": [
                c.name for c in spec.columns if c.available_at == "identifier"
            ],
            "leakage_rule": (
                f"a feature may only be derived from columns with available_at == "
                f"'{spec.decision_stage}'; identifiers and later-stage columns are excluded"
            ),
        },
        **narrative,
        "privacy": dict(spec.privacy),
    }
