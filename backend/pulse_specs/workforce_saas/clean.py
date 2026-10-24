"""The clean step of workforce_saas: thirteen raw tables -> one row per account and calendar month.

``clean_tables`` first normalises the raw export (exact duplicates, enum spelling, swapped date
ranges; every fix is counted in ``report["raw_fixes"]``), then builds the account-month spine
from ``accounts`` and aggregates each data group onto it with one ``groupby`` per table. Every
column tagged ``month_end`` in ``columns.py`` uses only events up to the last second of the month;
``cancelled_at`` is read once, for the post-decision columns and the target.

The funnel (``report["rules"]``) works at the account-month grain: months whose outcome is not
known at the snapshot, the cancellation month itself, and months without a billing row are
dropped. ``accounts.status`` and the ``truth/`` folder are never read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pulse_specs.workforce_saas import columns as wf

HORIZON = pd.Timedelta(days=wf.HORIZON_DAYS)
INVOICE_DUE_DAYS = 14
KEY = ["account_id", "period"]
ENUM_COLUMNS = {
    "punches": ("method", "direction"),
    "shifts": ("outcome",),
    "absence_requests": ("kind", "decision"),
}
ENUM_ALIASES = {"method": {"mobile": "app"}}
RANGE_COLUMNS = {
    "absence_requests": ("start_date", "end_date"),
    "shifts": ("planned_start", "planned_end"),
}
DEFINITIONS = {
    wf.TARGET: "cancelled_at in (month_end, month_end + 60 days]; equals 1 <= days_to_cancel <= 60",
    "days_to_cancel": "ceil of the seconds between month_end and cancelled_at, in days",
    "in_trial": "the trial was still running at the start of the month (trial_ends_at >= month_start)",
    "payroll_exported": "the previous calendar month was exported to payroll by this month end",
    "invoice_days_late": "days between the due date (period + 14 days) and the payment, when paid by month end",
    "payment_missing": "the invoice was positive and unpaid at the month end",
    "data_gap": "device offline hours above half of the device-hours in the month",
    "missing out punches": "need no fix: hours come from hours_daily, punches count every row",
}


# ----------------------------------------------------------------------------- raw normalisation


def normalise_raw(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Exact duplicates, enum spelling and swapped ranges, each counted per table."""
    fixed = {}
    fixes: dict[str, dict] = {"duplicate_rows": {}, "bad_enum": {}, "swapped_range": {}}
    for name, df in tables.items():
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        fixes["duplicate_rows"][name] = int(before - len(df))
        if name in ENUM_COLUMNS:
            changed = 0
            for column in ENUM_COLUMNS[name]:
                original = df[column]
                cleaned = original.str.strip().str.lower().replace(ENUM_ALIASES.get(column, {}))
                changed += int(((cleaned != original) & original.notna()).sum())
                df[column] = cleaned
            fixes["bad_enum"][name] = changed
        if name in RANGE_COLUMNS:
            first, second = RANGE_COLUMNS[name]
            swapped = (df[second] < df[first]).fillna(False).to_numpy()
            df.loc[swapped, [first, second]] = df.loc[swapped, [second, first]].to_numpy()
            fixes["swapped_range"][name] = int(swapped.sum())
        fixed[name] = df
    fixes["total"] = int(sum(sum(v.values()) for v in fixes.values()))
    return fixed, fixes


# ----------------------------------------------------------------------------- the spine


def _period(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M")


def build_spine(accounts: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """One row per account and calendar month, from the sign-up month to the last month alive."""
    acc = accounts[
        ["account_id", "industry", "region", "signed_up_at", "trial_ends_at", "cancelled_at"]
    ]
    acc = acc.sort_values("account_id", kind="mergesort").reset_index(drop=True)
    start = _period(acc["signed_up_at"])
    end = _period(acc["cancelled_at"].fillna(snapshot - pd.Timedelta(seconds=1)))
    months = (end.dt.year * 12 + end.dt.month) - (start.dt.year * 12 + start.dt.month) + 1
    months = months.clip(lower=1).to_numpy()
    offset = np.arange(months.sum()) - np.repeat(np.cumsum(months) - months, months)
    spine = acc.loc[acc.index.repeat(months)].reset_index(drop=True)
    spine["industry"] = spine["industry"].str.strip().str.lower()
    spine["region"] = spine["region"].str.strip().str.lower()
    spine["period"] = start.repeat(months).reset_index(drop=True) + offset
    spine["tenure_months"] = offset + 1
    spine["month"] = spine["period"].astype(str)
    spine["month_start"] = spine["period"].dt.start_time
    spine["month_end"] = spine["period"].dt.end_time.dt.floor("s")
    spine["days_in_month"] = spine["period"].dt.days_in_month
    spine[wf.ID_COLUMN] = spine["account_id"] + ":" + spine["month"]
    return spine


# ----------------------------------------------------------------------------- the groups


def _in_month(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Copy with a ``period`` key from ``column``; rows without a timestamp are dropped."""
    out = df[df[column].notna()].copy()
    out["period"] = _period(out[column])
    return out


def _with_month_end(df: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    return df.merge(spine[[*KEY, "month_end"]], on=KEY, how="inner")


def _commercial(spine: pd.DataFrame, billing: pd.DataFrame) -> pd.DataFrame:
    b = _in_month(billing, "period")[[*KEY, "plan", "seats", "mrr", "paid_at"]]
    b = b.drop_duplicates(KEY, keep="first")
    out = spine[[*KEY, "month_start", "month_end", "trial_ends_at"]].merge(b, on=KEY, how="left")
    out["in_trial"] = out["trial_ends_at"] >= out["month_start"]
    due = out["month_start"] + pd.Timedelta(days=INVOICE_DUE_DAYS)
    paid_in_month = out["paid_at"].notna() & (out["paid_at"] <= out["month_end"]) & (out["mrr"] > 0)
    late = np.floor((out["paid_at"] - due).dt.total_seconds() / 86400).clip(lower=0)
    out["invoice_days_late"] = pd.array(late.where(paid_in_month), dtype="Int64")
    out["payment_missing"] = (out["mrr"] > 0) & (
        out["paid_at"].isna() | (out["paid_at"] > out["month_end"])
    )
    out["has_billing"] = out["plan"].notna()
    return out[
        [
            *KEY,
            "plan",
            "seats",
            "mrr",
            "in_trial",
            "invoice_days_late",
            "payment_missing",
            "has_billing",
        ]
    ]


def _organisation(employees: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    joined = _in_month(employees, "hired_at").groupby(KEY).size().rename("employees_joined")
    left = _in_month(employees, "left_at").groupby(KEY).size().rename("employees_left")
    closed = _in_month(sites, "closed_at").groupby(KEY).size().rename("sites_closed")
    return pd.concat([joined, left, closed], axis=1).reset_index()


def _punches(punches: pd.DataFrame) -> pd.DataFrame:
    p = _in_month(punches, "punched_at")
    p["day"] = p["punched_at"].dt.normalize()
    p["is_app"] = p["method"] == "app"
    p["is_edited"] = p["edited"].fillna(False).astype(bool)
    g = p.groupby(KEY)
    out = pd.DataFrame(
        {
            "punches": g.size(),
            "punch_days": g["day"].nunique(),
            "active_employees": g["employee_id"].nunique(),
            "active_sites": g["site_id"].nunique(),
            "app_punches": g["is_app"].sum(),
            "edited_punches": g["is_edited"].sum(),
        }
    ).reset_index()
    out["app_punch_share"] = out["app_punches"] / out["punches"]
    out["edited_punch_share"] = out["edited_punches"] / out["punches"]
    return out.drop(columns=["app_punches", "edited_punches"])


def _shifts(shifts: pd.DataFrame) -> pd.DataFrame:
    s = _in_month(shifts, "planned_start")
    s["is_open"] = s["outcome"] == "open"
    s["is_no_show"] = s["outcome"] == "no_show"
    s["lead_days"] = ((s["planned_start"] - s["published_at"]).dt.total_seconds() / 86400).clip(
        lower=0
    )
    g = s.groupby(KEY)
    out = pd.DataFrame(
        {
            "shifts_planned": g.size(),
            "open_shifts": g["is_open"].sum(),
            "no_shows": g["is_no_show"].sum(),
            "schedule_lead_days": g["lead_days"].mean(),
        }
    ).reset_index()
    assigned = out["shifts_planned"] - out["open_shifts"]
    out["no_show_rate"] = (out["no_shows"] / assigned).where(assigned > 0)
    out["open_shift_share"] = out["open_shifts"] / out["shifts_planned"]
    return out.drop(columns=["open_shifts", "no_shows"])


def _absences(absences: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    filed = _with_month_end(_in_month(absences, "requested_at"), spine)
    decided = filed["decided_at"].notna() & (filed["decided_at"] <= filed["month_end"])
    resolved = filed["decided_at"].where(decided, filed["month_end"])
    filed["wait_days"] = ((resolved - filed["requested_at"]).dt.total_seconds() / 86400).clip(
        lower=0
    )
    g = filed.groupby(KEY)
    requests = pd.DataFrame(
        {"absence_requests": g.size(), "pending_request_days": g["wait_days"].mean()}
    )

    sick = _with_month_end(_in_month(absences, "start_date"), spine)
    sick = sick[
        (sick["kind"] == "sick")
        & (sick["decision"] == "approved")
        & sick["decided_at"].notna()
        & (sick["decided_at"] <= sick["month_end"])
    ]
    sick_days = ((sick["end_date"] - sick["start_date"]).dt.days + 1).clip(lower=1)
    sick_days = sick_days.groupby([sick["account_id"], sick["period"]]).sum().rename("sick_days")
    return pd.concat([requests, sick_days], axis=1).reset_index()


def _hours(hours: pd.DataFrame, wages: pd.DataFrame | None) -> pd.DataFrame:
    """Overtime and rest violations; labor_cost only when the wages (group 10) are there too."""
    h = _in_month(hours, "work_date")
    for column in ("regular_hours", "overtime_125_hours", "overtime_150_hours"):
        h[column] = h[column].fillna(0.0)
    h["overtime"] = h["overtime_125_hours"] + h["overtime_150_hours"]
    h["all_hours"] = h["regular_hours"] + h["overtime"]
    h["violation"] = h["rest_violation"].fillna(False).astype(bool)
    h = h.sort_values(["work_date", "account_id", "employee_id"], kind="mergesort")
    if wages is not None:
        w = wages[wages["effective_from"].notna() & wages["hourly_rate"].notna()]
        w = w.sort_values(["effective_from", "account_id", "employee_id"], kind="mergesort")
        h = pd.merge_asof(
            h,
            w[["account_id", "employee_id", "effective_from", "hourly_rate"]],
            left_on="work_date",
            right_on="effective_from",
            by=["account_id", "employee_id"],
            direction="backward",
        )
        h["cost"] = h["all_hours"] * h["hourly_rate"]
    g = h.groupby(KEY)
    parts = {
        "overtime_hours": g["overtime"].sum(),
        "all_hours": g["all_hours"].sum(),
        "rest_violations": g["violation"].sum(),
    }
    if wages is not None:
        parts["labor_cost"] = g["cost"].sum(min_count=1)
    out = pd.DataFrame(parts).reset_index()
    out["overtime_share"] = (out["overtime_hours"] / out["all_hours"]).where(
        out["all_hours"] > 0, 0.0
    )
    return out.drop(columns=["overtime_hours", "all_hours"])


def _payroll(exports: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    e = exports[exports["period"].notna() & exports["exported_at"].notna()].copy()
    e["period"] = _period(e["period"]) + 1  # the export of period P counts for month P + 1
    e = _with_month_end(e, spine)
    e = e[e["exported_at"] <= e["month_end"]]
    e = e.sort_values(["account_id", "period", "exported_at"], kind="mergesort").drop_duplicates(
        KEY, keep="first"
    )
    month_start = e["period"].dt.start_time
    e["payroll_export_delay_days"] = np.floor(
        (e["exported_at"] - month_start).dt.total_seconds() / 86400
    )
    e["payroll_exported"] = True
    e["payroll_corrections"] = e["corrections_after_export"].fillna(0)
    return e[[*KEY, "payroll_exported", "payroll_export_delay_days", "payroll_corrections"]]


def _usage(usage: pd.DataFrame) -> pd.DataFrame:
    u = _in_month(usage, "usage_date").sort_values(["account_id", "usage_date"], kind="mergesort")
    u["admin_day"] = u["admin_logins"].fillna(0) > 0
    flags = ["scheduling_active", "absences_active", "documents_active"]
    for column in flags:
        u[column] = u[column].fillna(False).astype(bool)
    g = u.groupby(KEY)
    last = g[[*flags, "app_installs_total"]].last()
    out = pd.DataFrame(
        {
            "admin_login_days": g["admin_day"].sum(),
            "reports_viewed": g["reports_viewed"].sum(),
            "messages_sent": g["messages_sent"].sum(),
            "support_tickets": g["support_tickets"].sum(),
            "modules_active": last[flags].sum(axis=1),
            "app_installs_total": last["app_installs_total"],
        }
    )
    return out.reset_index()


def _devices(devices: pd.DataFrame, health: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    d = spine[[*KEY, "month_end"]].merge(
        devices[["account_id", "installed_at", "retired_at"]], on="account_id", how="inner"
    )
    installed = (d["installed_at"] <= d["month_end"]) & (
        d["retired_at"].isna() | (d["retired_at"] > d["month_end"])
    )
    total = d[installed].groupby(KEY).size().rename("devices_total")
    h = _in_month(health, "health_date").groupby(KEY)
    frame = pd.DataFrame(
        {
            "devices_total": total,
            "device_offline_hours": h["offline_hours"].sum(),
            "failed_recognitions": h["failed_recognitions"].sum(),
        }
    )
    return frame.reset_index()


# ----------------------------------------------------------------------------- assembly


def _parts(
    spine: pd.DataFrame, tables: dict[str, pd.DataFrame], groups: tuple[str, ...]
) -> list[pd.DataFrame]:
    """One aggregate per data group that is present, in a fixed order."""
    t = tables
    parts = [_commercial(spine, t["billing_monthly"])]
    if "organisation" in groups:
        parts.append(_organisation(t["employees"], t["sites"]))
    parts.append(_punches(t["punches"]))
    if "shifts" in groups:
        parts.append(_shifts(t["shifts"]))
    if "absences" in groups:
        parts.append(_absences(t["absence_requests"], spine))
    if "hours" in groups:
        parts.append(_hours(t["hours_daily"], t["wages"] if "money" in groups else None))
    parts.append(_payroll(t["payroll_exports"], spine))
    parts.append(_usage(t["usage_daily"]))
    if "devices" in groups:
        parts.append(_devices(t["devices"], t["device_health_daily"], spine))
    return parts


def _merge_all(
    spine: pd.DataFrame, tables: dict[str, pd.DataFrame], groups: tuple[str, ...]
) -> pd.DataFrame:
    df = spine
    for part in _parts(spine, tables, groups):
        df = df.merge(part, on=KEY, how="left")
    zero = [
        "employees_joined",
        "employees_left",
        "sites_closed",
        "punches",
        "punch_days",
        "active_employees",
        "active_sites",
        "app_punch_share",
        "edited_punch_share",
        "shifts_planned",
        "absence_requests",
        "sick_days",
        "rest_violations",
        "overtime_share",
        "payroll_corrections",
        "admin_login_days",
        "reports_viewed",
        "messages_sent",
        "support_tickets",
        "modules_active",
        "app_installs_total",
        "devices_total",
        "device_offline_hours",
        "failed_recognitions",
    ]
    for column in zero:
        if column in df:
            df[column] = df[column].fillna(0)
    df["payroll_exported"] = df["payroll_exported"].eq(True)
    df["has_billing"] = df["has_billing"].eq(True)
    active = df["active_employees"].astype(float)
    if "absences" in groups:
        df["sick_days_per_employee"] = (df["sick_days"] / active).where(active > 0, 0.0)
    df["app_install_share"] = (df["app_installs_total"] / active).clip(0, 1).where(active > 0)
    if "devices" in groups:
        punches = df["punches"].astype(float)
        failures = df["failed_recognitions"] / punches
        df["device_failure_rate"] = failures.clip(0, 1).where(punches > 0)
        capacity = 0.5 * 24 * df["days_in_month"] * df["devices_total"]
        df["data_gap"] = (df["devices_total"] > 0) & (df["device_offline_hours"] > capacity)
    seconds = (df["cancelled_at"] - df["month_end"]).dt.total_seconds()
    df["days_to_cancel"] = pd.array(np.ceil(seconds / 86400), dtype="Int64")
    df[wf.TARGET] = df["days_to_cancel"].between(1, wf.HORIZON_DAYS).fillna(False).astype(bool)
    return df


def _typed(df: pd.DataFrame, groups: tuple[str, ...]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for column in wf.columns_for(groups):
        values = df[column.name]
        if column.dtype == "int":
            out[column.name] = (
                pd.array(values, dtype="Int64") if column.nullable else values.astype("int64")
            )
        elif column.dtype == "bool":
            out[column.name] = values.astype(bool)
        elif column.dtype == "float":
            out[column.name] = values.astype("float64").round(4)
        elif column.dtype == "datetime":
            out[column.name] = pd.to_datetime(values)
        else:
            out[column.name] = values.astype(object)
    return out


def _rule(name: str, description: str, before: int, after: int, detail: str) -> dict:
    return {
        "rule": name,
        "description": description,
        "rows_before": int(before),
        "rows_after": int(after),
        "rows_removed": int(before - after),
        "detail": detail,
    }


def clean_tables(
    tables: dict[str, pd.DataFrame],
    lookups_dir: str | Path,
    *,
    snapshot: pd.Timestamp,
    groups: tuple[str, ...] = wf.ALL_GROUPS,
) -> tuple[pd.DataFrame, dict]:
    """The DatasetSpec ``clean`` callable, bound to a snapshot and the data groups by make_spec.

    Tables of groups outside ``groups`` are never read, even when they are in ``tables``.
    """
    del lookups_dir  # the spec needs no lookups
    needed = {t for g in groups for t in wf.GROUP_TABLES[g]}
    tables = {name: df for name, df in tables.items() if name in needed}
    tables, raw_fixes = normalise_raw(tables)
    spine = build_spine(tables["accounts"], snapshot)
    rows_in = len(spine)
    rules = []

    known = spine["month_end"] + HORIZON <= snapshot
    kept = spine[known]
    rules.append(
        _rule(
            "R1_drop_censored_months",
            "months whose outcome is not known at the snapshot",
            rows_in,
            len(kept),
            f"month_end + {wf.HORIZON_DAYS} days must not be after the snapshot {snapshot.date()}",
        )
    )
    before = len(kept)
    kept = kept[kept["cancelled_at"].isna() | (kept["cancelled_at"] > kept["month_end"])]
    rules.append(
        _rule(
            "R2_drop_cancellation_month",
            "the month in which the account cancelled",
            before,
            len(kept),
            "the account is no longer a customer at the decision time, so days_to_cancel >= 1",
        )
    )
    df = _merge_all(kept.reset_index(drop=True), tables, groups)
    before = len(df)
    df = df[df["has_billing"]]
    rules.append(
        _rule(
            "R3_drop_months_without_billing",
            "months without a billing_monthly row",
            before,
            len(df),
            "plan, seats and mrr come from the bill and are not imputed",
        )
    )
    df = df.sort_values([wf.TIME_COLUMN, "account_id"], kind="mergesort").reset_index(drop=True)
    clean = _typed(df, groups)
    report = {
        "rows_in": int(rows_in),
        "rows_out": int(len(clean)),
        "rules": rules,
        "definitions": dict(DEFINITIONS),
        "raw_fixes": raw_fixes,
    }
    return clean, report
