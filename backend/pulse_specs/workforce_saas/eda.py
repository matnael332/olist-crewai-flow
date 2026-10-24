"""Exploratory summary and figures of the workforce_saas clean table."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pulse_specs.workforce_saas import columns as wf  # noqa: E402

TENURE_BUCKETS = ((1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 12, "7-12"), (13, 10**6, "13+"))
BLUE, RED = "#2563eb", "#dc2626"


def _bucket(tenure: pd.Series) -> pd.Series:
    labels = pd.Series("", index=tenure.index, dtype=object)
    for low, high, label in TENURE_BUCKETS:
        labels[(tenure >= low) & (tenure <= high)] = label
    return labels


def _rate_by(df: pd.DataFrame, key: pd.Series, target: pd.Series) -> dict:
    grouped = target.groupby(key)
    return {
        str(k): {"rows": int(n), "positive_rate": round(float(r), 4)}
        for (k, n), r in zip(grouped.size().items(), grouped.mean(), strict=True)
    }


def eda_summary(tables: dict[str, pd.DataFrame], clean: pd.DataFrame, report: dict) -> dict:
    target = clean[wf.TARGET].astype(bool)
    rate = float(target.mean())
    exported = float(clean["payroll_exported"].astype(bool).mean())
    cancelled = int(tables["accounts"]["cancelled_at"].notna().sum())
    raw_fixes = report.get("raw_fixes", {})
    nullable = [
        c.name
        for c in wf.COLUMNS
        if c.nullable and c.available_at == wf.DECISION_STAGE and c.name in clean
    ]
    return {
        "positive_rate": rate,
        "key_numbers": [
            {"label": "accounts", "value": int(clean["account_id"].nunique())},
            {"label": "account-months", "value": int(len(clean))},
            {"label": "cancelled accounts", "value": cancelled},
            {"label": "positive rate", "value": rate},
            {"label": "months with payroll export share", "value": exported},
            {"label": "raw rows fixed", "value": int(raw_fixes.get("total", 0))},
        ],
        "positive_rate_by_industry": _rate_by(clean, clean["industry"], target),
        "positive_rate_by_plan": _rate_by(clean, clean["plan"], target),
        "positive_rate_by_in_trial": _rate_by(clean, clean["in_trial"].astype(bool), target),
        "positive_rate_by_tenure_bucket": _rate_by(
            clean, _bucket(clean["tenure_months"].astype(int)), target
        ),
        "monthly_positive_rate": _rate_by(clean, clean["month"], target),
        "raw_fixes": raw_fixes,
        "cleaning_funnel": {r["rule"]: r["rows_after"] for r in report.get("rules", [])},
        "null_rates": {name: round(float(clean[name].isna().mean()), 4) for name in nullable},
    }


def eda_figures(summary: dict, clean: pd.DataFrame, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    names = []

    by_tenure = summary["positive_rate_by_tenure_bucket"]
    order = [label for _lo, _hi, label in TENURE_BUCKETS if label in by_tenure]
    fig = Figure(figsize=(5, 3))
    ax = fig.subplots()
    ax.bar(order, [by_tenure[k]["positive_rate"] for k in order], color=BLUE)
    ax.set_title("churned_60d rate by tenure (months)")
    ax.set_ylabel("positive rate")
    fig.tight_layout()
    fig.savefig(out / "positive_rate_by_tenure.png", dpi=80, metadata={"Software": None})
    names.append("positive_rate_by_tenure.png")

    target = clean[wf.TARGET].astype(bool)
    logins = clean["admin_login_days"].astype(int)
    fig = Figure(figsize=(5, 3))
    ax = fig.subplots()
    bins = np.arange(0, 33, 2)
    ax.hist(logins[~target], bins=bins, color=BLUE, alpha=0.7, density=True, label="stayed")
    ax.hist(logins[target], bins=bins, color=RED, alpha=0.7, density=True, label="churned")
    ax.set_title("admin login days in the month")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "admin_login_days_by_outcome.png", dpi=80, metadata={"Software": None})
    names.append("admin_login_days_by_outcome.png")

    monthly = summary["monthly_positive_rate"]
    months = sorted(monthly)
    fig = Figure(figsize=(7, 3))
    ax = fig.subplots()
    ax.plot(
        months, [monthly[m]["positive_rate"] for m in months], color=RED, marker="o", markersize=3
    )
    ax.set_ylabel("positive rate")
    ax.set_title("churned_60d rate per month")
    ax.set_xticks(months[:: max(1, len(months) // 8)])
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "monthly_positive_rate.png", dpi=80, metadata={"Software": None})
    names.append("monthly_positive_rate.png")
    return names
