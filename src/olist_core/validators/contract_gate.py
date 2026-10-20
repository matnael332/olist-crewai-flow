"""Gate 1 (handoff): does clean_data.csv honor dataset_contract.json?

Reads the CSV as strings and converts every column with the contract's own dtype, so a column that
does not parse as its declared type is reported, not silently coerced. Never raises.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from olist_core.columns import parse_column, read_csv_strings
from olist_core.io import read_json
from olist_core.validators.report import GateReport

GATE = "gate_contract"


def _to_numpy_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in out.columns:
        dtype = str(out[name].dtype)
        if dtype == "Int64" and not out[name].isna().any():
            out[name] = out[name].astype("int64")
        elif dtype == "boolean" and not out[name].isna().any():
            out[name] = out[name].astype(bool)
    return out


def _violations(df: pd.DataFrame, constraint: dict) -> int:
    kind = constraint["type"]
    if kind == "unique":
        return int(df.duplicated(subset=constraint["columns"]).sum())
    if kind == "date_range":
        col = df[constraint["column"]]
        lo = pd.Timestamp(constraint["min"])
        hi = pd.Timestamp(constraint["max_exclusive"])
        return int(((col < lo) | (col >= hi) | col.isna()).sum())
    if kind == "query":
        subset = df
        if constraint.get("where_not_null"):
            subset = df.dropna(subset=constraint["where_not_null"])
        subset = _to_numpy_dtypes(subset)
        result = subset.eval(constraint["query"], engine="python")
        result = pd.Series(result, index=subset.index).fillna(False).astype(bool)
        return int((~result).sum())
    raise ValueError(f"unknown constraint type {kind!r}")


def validate_clean_data_against_contract(
    clean_csv: str | Path, contract_json: str | Path
) -> GateReport:
    report = GateReport(GATE)
    try:
        contract = read_json(contract_json)
        report.record("contract_readable", True, "dataset_contract.json parsed")
    except (OSError, ValueError) as exc:
        report.record("contract_readable", False, f"cannot read dataset_contract.json: {exc}")
        return report
    try:
        raw = read_csv_strings(clean_csv)
        report.record("data_readable", True, f"clean_data.csv loaded ({len(raw):,} rows)")
    except (OSError, ValueError) as exc:
        report.record("data_readable", False, f"cannot read clean_data.csv: {exc}")
        return report

    specs = {c["name"]: c for c in contract.get("columns", [])}
    missing = [c for c in specs if c not in raw.columns]
    extra = [c for c in raw.columns if c not in specs]
    report.record(
        "columns_match",
        not missing and not extra,
        "all contract columns present, no extras"
        if not missing and not extra
        else f"missing columns {missing}, unexpected columns {extra}",
    )

    typed, parse_errors = {}, []
    for name, spec in specs.items():
        if name not in raw.columns:
            continue
        typed[name], bad = parse_column(raw[name], spec["dtype"])
        if bad:
            parse_errors.append(f"{name}: {bad} values are not {spec['dtype']}")
    df = pd.DataFrame(typed, index=raw.index)
    report.record(
        "dtypes",
        not parse_errors,
        "every column parses as its declared dtype"
        if not parse_errors
        else f"dtype violations: {parse_errors}",
    )

    nulls = []
    for name, spec in specs.items():
        if name in df and not spec.get("nullable", False):
            count = int(df[name].isna().sum())
            if count:
                nulls.append(f"{name} ({count} nulls)")
    report.record(
        "non_nullable",
        not nulls,
        "no nulls in non-nullable columns" if not nulls else f"nulls in {nulls}",
    )

    out_of_bounds = []
    for name, spec in specs.items():
        bounds = spec.get("bounds")
        if not bounds or name not in df:
            continue
        values = df[name].dropna().astype(float)
        if "min" in bounds and (low := int((values < bounds["min"]).sum())):
            out_of_bounds.append(f"{name}: {low} below {bounds['min']}")
        if "max" in bounds and (high := int((values > bounds["max"]).sum())):
            out_of_bounds.append(f"{name}: {high} above {bounds['max']}")
    report.record(
        "bounds",
        not out_of_bounds,
        "all values within declared bounds"
        if not out_of_bounds
        else f"bound violations: {out_of_bounds}",
    )

    disallowed = []
    for name, spec in specs.items():
        allowed = spec.get("allowed_values")
        if allowed is None or name not in df:
            continue
        values = df[name].dropna()
        if spec["dtype"] == "bool":
            values = values.astype(bool)
        bad = sorted({str(v) for v in values.unique()} - {str(v) for v in allowed})
        if bad:
            disallowed.append(f"{name}: {bad[:5]}")
    report.record(
        "allowed_values",
        not disallowed,
        "all categorical values allowed" if not disallowed else f"disallowed values: {disallowed}",
    )

    pattern_errors = []
    for name, spec in specs.items():
        pattern = spec.get("pattern")
        if not pattern or name not in df:
            continue
        values = df[name].dropna().astype(str)
        bad = int((~values.str.fullmatch(re.compile(pattern))).sum())
        if bad:
            pattern_errors.append(f"{name}: {bad} values do not match {pattern}")
    report.record(
        "patterns",
        not pattern_errors,
        "all patterns match" if not pattern_errors else str(pattern_errors),
    )

    expected_rows = contract.get("row_count")
    report.record(
        "row_count",
        expected_rows == len(df),
        f"{len(df):,} rows (contract {expected_rows:,})"
        if isinstance(expected_rows, int)
        else "contract has no row_count",
    )

    date_range = contract.get("date_range", {})
    col = date_range.get("column")
    if col in df:
        lo, hi = pd.Timestamp(date_range["min"]), pd.Timestamp(date_range["max"])
        outside = int(((df[col] < lo) | (df[col] > hi)).sum())
        report.record(
            "date_range",
            outside == 0,
            f"{col} within {date_range['min']}..{date_range['max']}"
            if outside == 0
            else f"{outside} rows outside {date_range['min']}..{date_range['max']}",
        )

    for constraint in contract.get("constraints", []):
        try:
            bad = _violations(df, constraint)
            detail = f"{constraint['id']} {constraint.get('description', '')}: " + (
                "ok" if bad == 0 else f"{bad} violating rows"
            )
            report.record(f"constraint_{constraint['id']}", bad == 0, detail)
        except Exception as exc:  # noqa: BLE001 - a broken constraint is a failed check
            report.record(
                f"constraint_{constraint['id']}",
                False,
                f"{constraint['id']} cannot be evaluated: {exc}",
            )
    return report
