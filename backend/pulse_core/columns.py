"""The clean_data.csv column model and the typed readers shared by producers and gates.

``available_at`` drives the leakage rules: only columns tagged with the spec's decision stage may
feed model features; "identifier" columns are never features; "label" is the target; any other tag
(for example "post_decision") is forbidden as a feature source.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # string | int | float | bool | datetime
    available_at: str  # identifier | <decision stage> | <later stage> | label
    nullable: bool
    description: str


def dtypes_of(columns: Iterable[Column]) -> dict[str, str]:
    return {c.name: c.dtype for c in columns}


def parse_column(series: pd.Series, dtype: str) -> tuple[pd.Series, int]:
    """Convert a string column to its logical dtype. Returns (typed, unparseable non-empty count).

    Integers and booleans use pandas nullable dtypes so that null violations stay visible.
    """
    empty = series.isna() | (series == "")
    if dtype == "string":
        return series.where(~empty, None).astype("object"), 0
    if dtype in ("int", "float"):
        numbers = pd.to_numeric(series.where(~empty), errors="coerce")
        bad = numbers.isna() & ~empty
        if dtype == "int":
            non_integral = numbers.notna() & (numbers != numbers.round())
            bad = bad | non_integral
            numbers = numbers.where(~non_integral)
            return numbers.round().astype("Int64"), int(bad.sum())
        return numbers.astype("float64"), int(bad.sum())
    if dtype == "bool":
        mapped = series.map({"True": True, "False": False})
        bad = mapped.isna() & ~empty
        return mapped.astype("boolean"), int(bad.sum())
    if dtype == "datetime":
        parsed = pd.to_datetime(series.where(~empty), format=DATETIME_FORMAT, errors="coerce")
        return parsed, int((parsed.isna() & ~empty).sum())
    raise ValueError(f"unknown logical dtype {dtype!r}")


def read_csv_strings(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def read_clean_data(path, dtypes: dict[str, str]) -> pd.DataFrame:
    """Read clean_data.csv with explicit logical dtypes (columns not listed stay strings)."""
    raw = read_csv_strings(path)
    typed = {}
    for name in raw.columns:
        typed[name], _ = parse_column(raw[name], dtypes.get(name, "string"))
    return pd.DataFrame(typed, index=raw.index)
