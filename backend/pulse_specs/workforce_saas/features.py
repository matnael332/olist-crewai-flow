"""Model features of workforce_saas: every month-end column as is, plus trends and streaks.

Level features are the clean columns known at the month end (bools as 0/1, strings as
categories). Trend features compare a month with the account's own recent past and are therefore
allowed: the previous months are decision-stage columns too. Streaks count consecutive months of
silence. Nothing here reads ``cancelled_at``, ``days_to_cancel`` or the target.

Every definition is a function of the column catalogue, so a customer without an optional data
group gets the same features minus the ones built on that group's columns.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from pulse_core.columns import Column
from pulse_specs.workforce_saas import columns as wf

TREND_COLUMNS = (
    "punches",
    "active_employees",
    "seats",
    "admin_login_days",
    "modules_active",
    "overtime_share",
    "payroll_exported",
)
STREAKS = {
    "months_without_payroll_export": (
        "payroll_exported",
        "consecutive months without a payroll export",
    ),
    "months_without_admin_login": ("admin_login_days", "consecutive months without an admin login"),
}
REQUIRED = (
    "tenure_months",
    "seats",
    "punches",
    "active_employees",
    "admin_login_days",
    "modules_active",
    "payroll_exported",
    "plan",
    "industry",
)


def level_columns(columns: Sequence[Column] = wf.COLUMNS) -> tuple[Column, ...]:
    return tuple(
        c for c in columns if c.available_at == wf.DECISION_STAGE and c.name != wf.TIME_COLUMN
    )


def trend_columns(columns: Sequence[Column] = wf.COLUMNS) -> tuple[str, ...]:
    names = {c.name for c in columns}
    return tuple(c for c in TREND_COLUMNS if c in names)


def _level_definitions(columns: Sequence[Column]) -> list[dict]:
    definitions = []
    for column in level_columns(columns):
        kind = "categorical" if column.dtype == "string" else "numeric"
        description = column.description + (" (0/1)" if column.dtype == "bool" else "")
        definitions.append(
            {
                "name": column.name,
                "kind": kind,
                "derived_from": [column.name],
                "nullable": column.nullable,
                "description": description,
            }
        )
    return definitions


def _trend_definitions(columns: Sequence[Column]) -> list[dict]:
    definitions = []
    for column in trend_columns(columns):
        for suffix, text in (
            ("1m", "the previous month"),
            ("3m", "the mean of the previous three months"),
        ):
            definitions.append(
                {
                    "name": f"{column}_delta_{suffix}",
                    "kind": "numeric",
                    "derived_from": [column],
                    "nullable": True,
                    "description": f"{column} minus its value in {text} (null in the account's first month)",
                }
            )
    return definitions


def _streak_definitions() -> list[dict]:
    return [
        {
            "name": name,
            "kind": "numeric",
            "derived_from": [source],
            "nullable": False,
            "description": f"{description}, up to and including this month",
        }
        for name, (source, description) in STREAKS.items()
    ]


def feature_definitions(columns: Sequence[Column] = wf.COLUMNS) -> tuple[dict, ...]:
    """Level, trend and streak features of the columns at hand (streaks use core columns only)."""
    return tuple(_level_definitions(columns) + _trend_definitions(columns) + _streak_definitions())


FEATURE_DEFINITIONS = feature_definitions()
REQUIRED_FEATURES: tuple[dict, ...] = tuple({"name": n, "derived_from": [n]} for n in REQUIRED)


def _streak(flag: pd.Series, account: pd.Series) -> pd.Series:
    """Length of the run of True values ending at each row, restarting per account and at False."""
    block = (~flag).groupby(account).cumsum()
    return flag.astype("int64").groupby([account, block]).cumsum()


def derive_features(df: pd.DataFrame, columns: Sequence[Column] = wf.COLUMNS) -> pd.DataFrame:
    """Features for every row of the clean table, index-aligned with it."""
    work = df.sort_values(["account_id", "month"], kind="mergesort")
    account = work["account_id"]
    out = pd.DataFrame(index=work.index)
    for column in level_columns(columns):
        values = work[column.name]
        if column.dtype == "string":
            out[column.name] = values.astype(object)
        elif column.dtype == "bool":
            out[column.name] = values.astype(bool).astype("int64")
        else:
            out[column.name] = values.astype("float64")
    for column in trend_columns(columns):
        x = out[column].astype("float64")
        previous = x.groupby(account).shift(1)
        mean3 = previous.groupby(account).transform(lambda s: s.rolling(3, min_periods=1).mean())
        out[f"{column}_delta_1m"] = (x - previous).round(4)
        out[f"{column}_delta_3m"] = (x - mean3).round(4)
    for name, (source, _description) in STREAKS.items():
        out[name] = _streak(out[source] == 0, account)
    return out.reindex(df.index)[[f["name"] for f in feature_definitions(columns)]]
