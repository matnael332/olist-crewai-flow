"""Time-based splits: day-granular windows, read from the spec (producers) or the contract (gates)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

EXCLUDED = "excluded"


def assign_split(ts: pd.Series, windows: Mapping[str, Sequence[str]]) -> pd.Series:
    """Window name per timestamp (inclusive day bounds); 'excluded' outside every window."""
    day = pd.to_datetime(ts).dt.floor("D")
    out = pd.Series(EXCLUDED, index=ts.index, dtype="object")
    for name, (lo, hi) in windows.items():
        mask = (day >= pd.Timestamp(lo)) & (day <= pd.Timestamp(hi))
        out = out.where(~mask, name)
    return out


def windows_of(split_policy: Mapping) -> dict[str, list[str]]:
    """The window entries of a contract's split_policy block (name -> [from, to])."""
    return {k: list(v) for k, v in split_policy.items() if isinstance(v, list | tuple)}


def split_policy(spec) -> dict:
    return {
        "type": "time",
        "column": spec.time_column,
        **{name: list(window) for name, window in spec.splits.items()},
        "cv": spec.cv_note or "TimeSeriesSplit inside the train window",
        "note": spec.split_note,
    }
