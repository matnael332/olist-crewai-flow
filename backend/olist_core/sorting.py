"""Stable sorting helpers.

numpy 2 dispatches SIMD quicksorts whose tie order differs between CPUs, so every sort that feeds an
artifact goes through a stable (mergesort) sort or a fully specified key.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def stable_sort(
    df: pd.DataFrame, by: str | Sequence[str], ascending: bool | Sequence[bool] = True
) -> pd.DataFrame:
    keys = [by] if isinstance(by, str) else list(by)
    return df.sort_values(by=keys, ascending=ascending, kind="mergesort").reset_index(drop=True)


def counts_sorted(series: pd.Series, top: int | None = None, name: str = "count") -> pd.DataFrame:
    """Value counts ordered by descending count, ties broken by the value itself."""
    counts = series.value_counts(dropna=False, sort=False)
    frame = pd.DataFrame({"value": counts.index.astype(str), name: counts.to_numpy()})
    frame = frame.sort_values(
        by=[name, "value"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    return frame.head(top) if top is not None else frame
