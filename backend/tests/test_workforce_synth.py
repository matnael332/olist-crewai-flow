"""The workforce_saas generator writes what the pipeline's input check expects, tells a consistent
story (keys, lifecycle, planted churn signals), is byte-reproducible and counts its own dirt."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pulse_core.io import sha256_file
from pulse_core.raw import (
    check_raw_schema,
    load_raw_schema,
    load_raw_tables,
    profile_raw_tables,
    verify_manifest,
)
from pulse_specs.workforce_saas import columns as wf
from pulse_specs.workforce_saas.synth import RAW_SCHEMA, SynthConfig, write_dataset
from pulse_specs.workforce_saas.synth_engine import PLAN_PRICE

TINY = SynthConfig(accounts=25, start_month="2023-01", snapshot_date="2024-07-01", dirt=0.0)
DIRTY = SynthConfig(accounts=25, start_month="2023-01", snapshot_date="2024-07-01", dirt=0.05)
MEDIUM = SynthConfig(accounts=80, start_month="2023-01", snapshot_date="2025-01-01", dirt=0.0)

ENUMS = {
    ("accounts", "industry"): set(wf.INDUSTRIES),
    ("accounts", "region"): set(wf.REGIONS),
    ("accounts", "status"): {"trial", "active", "cancelled"},
    ("billing_monthly", "plan"): set(wf.PLANS),
    ("punches", "direction"): {"in", "out"},
    ("punches", "method"): {"app", "biometric", "nfc", "web"},
    ("punches", "edited_by"): {"admin", "employee"},
    ("shifts", "outcome"): {"worked", "late", "no_show", "swapped", "open"},
    ("absence_requests", "kind"): {"sick", "vacation", "unpaid", "other"},
    ("absence_requests", "decision"): {"approved", "rejected"},
    ("devices", "device_type"): {"biometric", "nfc"},
}
REQUIRED_DATETIMES = {
    "accounts": ["signed_up_at", "trial_ends_at"],
    "billing_monthly": ["period"],
    "employees": ["hired_at"],
    "sites": ["opened_at"],
    "punches": ["punched_at"],
    "shifts": ["planned_start", "planned_end", "published_at"],
    "absence_requests": ["start_date", "end_date", "requested_at"],
    "hours_daily": ["work_date"],
    "payroll_exports": ["period", "exported_at"],
    "usage_daily": ["usage_date"],
    "devices": ["installed_at"],
    "device_health_daily": ["health_date"],
    "wages": ["effective_from"],
}
# Timestamps that must stay inside the account's life (before the cancellation and the snapshot).
EVENT_TIMES = {
    "billing_monthly": ["period"],
    "employees": ["hired_at", "left_at"],
    "sites": ["opened_at", "closed_at"],
    "punches": ["punched_at"],
    "shifts": ["planned_start", "published_at"],
    "absence_requests": ["requested_at", "decided_at"],
    "hours_daily": ["work_date"],
    "payroll_exports": ["period", "exported_at"],
    "usage_daily": ["usage_date"],
    "devices": ["installed_at", "retired_at"],
    "device_health_daily": ["health_date"],
    "wages": ["effective_from"],
}


def _dataset(tmp_path_factory, name: str, config: SynthConfig):
    data = tmp_path_factory.mktemp(name)
    raw = write_dataset(data, config)
    tables = load_raw_tables(raw)
    truth = {
        "accounts": pd.read_csv(data / "truth" / "accounts_truth.csv", keep_default_na=False),
        "monthly": pd.read_csv(data / "truth" / "monthly_health.csv"),
    }
    return data, tables, truth


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return _dataset(tmp_path_factory, "synth_tiny", TINY)


@pytest.fixture(scope="module")
def dirty(tmp_path_factory):
    return _dataset(tmp_path_factory, "synth_dirty", DIRTY)


@pytest.fixture(scope="module")
def medium(tmp_path_factory):
    return _dataset(tmp_path_factory, "synth_medium", MEDIUM)


def _life(tables: dict[str, pd.DataFrame], snapshot: str) -> pd.DataFrame:
    accounts = tables["accounts"].set_index("account_id")
    end = accounts["cancelled_at"].fillna(pd.Timestamp(snapshot))
    return pd.DataFrame({"signed_up_at": accounts["signed_up_at"], "end": end})


# ----------------------------------------------------------------------------- 1. the gate


def test_output_passes_the_pipeline_input_check(tiny):
    data, tables, _ = tiny
    raw = data / "raw"
    manifest = verify_manifest(raw)
    assert len(manifest["files"]) == 13 and "raw_schema.json" not in manifest["files"]
    assert manifest["generator"]["accounts"] == TINY.accounts
    report = check_raw_schema(raw)
    assert report.passed, report.errors
    assert len(report.checks) == 13
    assert set(tables) == set(RAW_SCHEMA["tables"])
    for name, df in tables.items():
        assert len(df) == manifest["files"][f"{name}.csv"]["rows"], name
    assert (data / "lookups").is_dir()
    assert (data / "truth" / "dirt_counts.json").read_text() == "{}\n"


# ----------------------------------------------------------------------------- 2. clean values


def test_clean_output_has_valid_keys_types_and_enums(tiny):
    _, tables, _ = tiny
    for name, columns in REQUIRED_DATETIMES.items():
        for column in columns:
            assert tables[name][column].notna().all(), f"{name}.{column}"
    profile = profile_raw_tables(tables, RAW_SCHEMA)
    for name, entry in profile.items():
        assert entry["duplicate_keys"] == 0, name
        assert entry["exact_duplicate_rows"] == 0, name
    for (name, column), allowed in ENUMS.items():
        seen = set(tables[name][column].dropna())
        assert seen <= allowed, (name, column, seen - allowed)
    punches = tables["punches"]
    assert (punches["device_id"].isna() == punches["method"].isin(["app", "web"])).all()
    assert (punches["edited_by"].isna() == ~punches["edited"]).all()
    hours = tables["hours_daily"]
    assert (hours["regular_hours"] <= 8.6).all() and (hours["overtime_125_hours"] <= 2).all()
    assert (hours[["regular_hours", "overtime_150_hours", "night_hours"]] >= 0).all().all()
    billing = tables["billing_monthly"]
    paid = billing[billing["mrr"] > 0]
    expected = paid["seats"] * paid["plan"].map(PLAN_PRICE)
    assert (paid["mrr"] == expected.astype(float)).all()
    assert (billing["paid_at"].isna() == billing["days_late"].isna()).all()


def test_referential_integrity(tiny):
    _, tables, _ = tiny
    parents = {
        "employee_id": tables["employees"],
        "site_id": tables["sites"],
        "device_id": tables["devices"],
    }
    for name, df in tables.items():
        for key, parent in parents.items():
            if key not in df.columns or parent is df:
                continue
            child = df.loc[df[key].notna(), ["account_id", key]].drop_duplicates()
            merged = child.merge(
                parent[["account_id", key]], on=["account_id", key], how="left", indicator=True
            )
            assert (merged["_merge"] == "both").all(), f"{name}.{key} points nowhere"
    devices = tables["devices"].merge(
        tables["sites"], on=["account_id", "site_id"], how="left", indicator=True
    )
    assert (devices["_merge"] == "both").all()
    accounts = set(tables["accounts"]["account_id"])
    for name, df in tables.items():
        assert set(df["account_id"]) <= accounts, name


# ----------------------------------------------------------------------------- 3. lifecycle


def test_lifecycle_is_consistent(tiny):
    _, tables, truth = tiny
    snapshot = pd.Timestamp(TINY.snapshot_date)
    accounts = tables["accounts"].set_index("account_id")
    cancelled = accounts.dropna(subset=["cancelled_at"])
    assert (cancelled["cancelled_at"] > cancelled["signed_up_at"]).all()
    assert (cancelled["cancelled_at"] < snapshot).all()
    reasons = truth["accounts"].set_index("account_id")["cancel_reason"]
    converted = cancelled[reasons.reindex(cancelled.index) != "trial_not_converted"]
    assert (converted["cancelled_at"] > converted["trial_ends_at"]).all()
    assert (cancelled["status"] == "cancelled").all()
    alive = accounts[accounts["cancelled_at"].isna()]
    assert (alive["status"] == np.where(alive["trial_ends_at"] > snapshot, "trial", "active")).all()

    life = _life(tables, TINY.snapshot_date)
    for name, columns in EVENT_TIMES.items():
        df = tables[name].join(life, on="account_id")
        for column in columns:
            stamped = df[df[column].notna()]
            assert (stamped[column] < stamped["end"]).all(), f"{name}.{column} after the end"
            assert (stamped[column] < snapshot).all(), f"{name}.{column} after the snapshot"

    billing = tables["billing_monthly"]
    for account_id, group in billing.groupby("account_id"):
        first = life.loc[account_id, "signed_up_at"].to_period("M")
        last = (life.loc[account_id, "end"] - pd.Timedelta(seconds=1)).to_period("M")
        expected = [p.start_time for p in pd.period_range(first, last, freq="M")]
        assert list(group["period"]) == expected, account_id
    assert truth["monthly"].groupby("account_id").size().sum() == len(billing)


# ----------------------------------------------------------------------------- 4. planted signals


def test_churn_rates_and_lead_signals(medium):
    _, tables, truth = medium
    snapshot = pd.Timestamp(MEDIUM.snapshot_date)
    monthly = truth["monthly"].copy()
    monthly["month_end"] = pd.PeriodIndex(monthly["month"], freq="M").end_time
    known = monthly[monthly["month_end"] + pd.Timedelta(days=60) <= snapshot]
    assert 0.03 <= known["churned_60d"].mean() <= 0.12, known["churned_60d"].mean()

    accounts = tables["accounts"].set_index("account_id")
    reasons = truth["accounts"].set_index("account_id")
    after_trial = monthly.join(accounts["trial_ends_at"], on="account_id")
    exposure = int((after_trial["month_end"] > after_trial["trial_ends_at"]).sum())
    converted_churn = int((reasons["cancel_reason"].isin(["random", "fade", "price"])).sum())
    assert 0.010 <= converted_churn / exposure <= 0.045, (converted_churn, exposure)

    by_archetype = reasons.groupby("archetype")["cancelled_at"].apply(lambda s: (s != "").mean())
    assert by_archetype["device_trouble"] <= by_archetype["slow_fade"]
    assert by_archetype["trial_bounce"] >= 0.8

    usage = tables["usage_daily"].copy()
    usage["month"] = usage["usage_date"].dt.to_period("M").astype(str)
    logins = usage.groupby(["account_id", "month"])["admin_logins"].sum()
    faders = reasons[(reasons["cancel_reason"] == "fade") & (reasons["months"] >= 7)].index
    assert len(faders) >= 2
    late, early = [], []
    for account_id in faders:
        series = logins.loc[account_id]
        late.append(series.iloc[-3:].mean())
        early.append(series.iloc[-6:-3].mean())
    assert np.mean(late) < 0.6 * np.mean(early), (late, early)

    punches = tables["punches"].copy()
    punches["month"] = punches["punched_at"].dt.to_period("M").astype(str)
    per_month = punches.groupby(["account_id", "month"]).size().rename("punches").reset_index()
    joined = monthly.merge(per_month, on=["account_id", "month"], how="left").fillna({"punches": 0})
    joined["relative"] = joined["punches"] / joined.groupby("account_id")["punches"].transform(
        "mean"
    ).replace(0, np.nan)
    assert joined[["relative", "health"]].corr().iloc[0, 1] > 0.15


# ----------------------------------------------------------------------------- 5. determinism


def _hashes(raw: Path) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(raw.iterdir())}


def test_same_config_is_byte_identical(tmp_path, tiny):
    first = _hashes(tiny[0] / "raw")
    again = write_dataset(tmp_path / "again", TINY)
    assert _hashes(again) == first
    other = write_dataset(
        tmp_path / "other", SynthConfig(**{**TINY.__dict__, "seed": TINY.seed + 1})
    )
    assert _hashes(other)["accounts.csv"] != first["accounts.csv"]


def test_more_accounts_only_append(tmp_path, tiny):
    bigger = write_dataset(
        tmp_path / "bigger", SynthConfig(**{**TINY.__dict__, "accounts": TINY.accounts + 5})
    )
    for name in ("accounts", "employees", "punches"):
        small = tiny[1][name]
        large = load_raw_tables(bigger, [name])[name]
        head = large[large["account_id"].isin(set(small["account_id"]))].reset_index(drop=True)
        pd.testing.assert_frame_equal(head, small)


def test_write_refuses_to_overwrite_without_force(tiny):
    with pytest.raises(FileExistsError):
        write_dataset(tiny[0], TINY)
    assert write_dataset(tiny[0], TINY, force=True) == tiny[0] / "raw"


# ----------------------------------------------------------------------------- 6. dirt


def test_dirt_is_planted_and_counted(tiny, dirty):
    _, clean, _ = tiny
    data, tables, _ = dirty
    counts = json.loads((data / "truth" / "dirt_counts.json").read_text())
    assert set(counts) == {
        "duplicate_rows",
        "missing_out",
        "null_non_key",
        "bad_enum",
        "swapped_range",
        "late_edit",
    }
    profile = profile_raw_tables(tables, load_raw_schema(data / "raw"))
    assert profile["punches"]["duplicate_keys"] > 0
    assert (
        profile["hours_daily"]["exact_duplicate_rows"]
        == counts["duplicate_rows"]["hours_daily"]
        > 0
    )
    expected = (
        len(clean["punches"])
        + counts["duplicate_rows"]["punches"]
        - counts["missing_out"]["punches"]
        + counts["late_edit"]["punches"]
    )
    assert len(tables["punches"]) == expected
    assert (tables["shifts"]["outcome"] == "WORKED").sum() == counts["bad_enum"]["shifts"] > 0
    assert (tables["absence_requests"]["kind"] == "Sick").sum() == counts["bad_enum"][
        "absence_requests"
    ]
    swapped = tables["shifts"]["planned_end"] < tables["shifts"]["planned_start"]
    assert swapped.sum() == counts["swapped_range"]["shifts"] > 0
    assert tables["employees"]["role"].isna().sum() > 0
    assert check_raw_schema(data / "raw").passed
    assert verify_manifest(data / "raw")


# ----------------------------------------------------------------------------- 7. the budget


@pytest.mark.slow
def test_default_config_fits_the_budget(tmp_path):
    started = time.perf_counter()
    raw = write_dataset(tmp_path, SynthConfig())
    elapsed = time.perf_counter() - started
    manifest = verify_manifest(raw)
    assert elapsed < 60, elapsed
    assert manifest["files"]["punches.csv"]["rows"] <= 1_000_000
    assert sum(entry["bytes"] for entry in manifest["files"].values()) < 160 * 1024 * 1024
    monthly = pd.read_csv(tmp_path / "truth" / "monthly_health.csv")
    assert monthly["churned_60d"].sum() >= 120
