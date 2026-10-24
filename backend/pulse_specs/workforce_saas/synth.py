"""Synthetic workforce_saas data: the thirteen raw tables, their manifest and the hidden truth.

    uv run synth --data-dir data            # default: 200 accounts, 2022-01 .. 2025-08
    uv run synth --accounts 300 --force     # bigger, overwriting data/raw

``generate`` builds every account from its own seeded generator (``synth_engine``), assigns the
public ids in a stable order and sprinkles a little realistic dirt. ``write_dataset`` writes
``<data_dir>/raw/`` exactly as the pipeline's input check expects it (tables, ``raw_schema.json``,
``manifest.json``), the hidden truth under ``<data_dir>/truth/`` and an empty ``lookups/``. Two
runs with the same config are byte-identical.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pulse_core.columns import DATETIME_FORMAT
from pulse_core.io import read_json, sha256_file, write_csv_stable, write_json_lf
from pulse_specs.workforce_saas.synth_engine import (
    AccountPlan,
    SynthConfig,
    build_account_tables,
    plan_account,
)

__all__ = ["SynthConfig", "apply_dirt", "format_for_csv", "generate", "main", "write_dataset"]

GENERATOR_VERSION = 1
SCHEMA_PATH = Path(__file__).with_name("raw_schema.json")
RAW_SCHEMA = read_json(SCHEMA_PATH)
TABLE_ORDER = tuple(RAW_SCHEMA["tables"])
DIRT_STREAM = 999_999
CSV_FLOATS = "%.2f"

# Stable sort keys per table (public column names); account first, so new accounts only append.
SORT_KEYS = {
    "accounts": ["account_id"],
    "billing_monthly": ["account_id", "period"],
    "employees": ["account_id", "hired_at", "_emp"],
    "sites": ["account_id", "opened_at", "_site"],
    "punches": ["account_id", "punched_at", "employee_id", "direction"],
    "shifts": ["account_id", "planned_start", "employee_id", "outcome"],
    "absence_requests": ["account_id", "requested_at", "employee_id", "start_date"],
    "hours_daily": ["account_id", "employee_id", "work_date"],
    "payroll_exports": ["account_id", "period"],
    "usage_daily": ["account_id", "usage_date"],
    "devices": ["account_id", "installed_at", "_dev"],
    "device_health_daily": ["account_id", "device_id", "health_date"],
    "wages": ["account_id", "employee_id", "effective_from"],
}
EVENT_IDS = {
    "punches": ("punch_id", "pun_{:08d}"),
    "shifts": ("shift_id", "shift_{:08d}"),
    "absence_requests": ("request_id", "req_{:07d}"),
    "payroll_exports": ("export_id", "exp_{:06d}"),
}


def _sorted(df: pd.DataFrame, table: str) -> pd.DataFrame:
    keys = [k for k in SORT_KEYS[table] if k in df.columns]
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _lookup(df: pd.DataFrame, local: str, ids: pd.Series) -> pd.Series:
    """Map (_acc, local index) pairs to the public id assigned to that row."""
    key = df["_acc"].to_numpy() * 1_000_000 + df[local].to_numpy()
    return pd.Series(ids.to_numpy(), index=key)


def _apply(df: pd.DataFrame, local: str, public: str, lookup: pd.Series) -> pd.DataFrame:
    key = df["_acc"].to_numpy() * 1_000_000 + df[local].to_numpy()
    mapped = lookup.reindex(key).to_numpy()
    df[public] = np.where(df[local].to_numpy() >= 0, mapped, "")
    return df


def _assign_ids(parts: dict[str, list[pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    tables = {}
    for name, frames in parts.items():
        filled = [frame for frame in frames if len(frame)] or frames[:1]
        tables[name] = pd.concat(filled, ignore_index=True)
    for df in tables.values():
        df["account_id"] = ["acc_" + format(int(a) + 1, "06d") for a in df["_acc"]]
    emp = _sorted(tables["employees"], "employees")
    emp["employee_id"] = ["emp_" + format(i, "06d") for i in range(1, len(emp) + 1)]
    sites = _sorted(tables["sites"], "sites")
    sites["site_id"] = ["site_" + format(i, "06d") for i in range(1, len(sites) + 1)]
    dev = _sorted(tables["devices"], "devices")
    dev["device_id"] = ["dev_" + format(i, "06d") for i in range(1, len(dev) + 1)]
    emp_ids = _lookup(emp, "_emp", emp["employee_id"])
    site_ids = _lookup(sites, "_site", sites["site_id"])
    dev_ids = _lookup(dev, "_dev", dev["device_id"])
    tables["employees"], tables["sites"], tables["devices"] = emp, sites, dev
    for name, df in tables.items():
        if "_emp" in df.columns and name != "employees":
            _apply(df, "_emp", "employee_id", emp_ids)
        if "_site" in df.columns and name != "sites":
            _apply(df, "_site", "site_id", site_ids)
        if "_dev" in df.columns and name != "devices":
            _apply(df, "_dev", "device_id", dev_ids)
    for name in tables:
        df = _sorted(tables[name], name)
        if name in EVENT_IDS:
            column, pattern = EVENT_IDS[name]
            df[column] = [pattern.format(i) for i in range(1, len(df) + 1)]
        tables[name] = df[list(RAW_SCHEMA["tables"][name]["columns"])]
    return tables


# ----------------------------------------------------------------------------- dirt

NULLABLE_STRINGS = {
    "employees": ["role", "employment_type"],
    "punches": ["method"],
    "devices": ["device_type"],
    "payroll_exports": ["payroll_software"],
    "absence_requests": ["kind"],
}
BAD_ENUMS = {
    "punches": ("method", {"app": "App", "web": "mobile"}),
    "shifts": ("outcome", {"worked": "WORKED"}),
    "absence_requests": ("kind", {"sick": "Sick"}),
}
SWAPPED_RANGES = {
    "absence_requests": ("start_date", "end_date"),
    "shifts": ("planned_start", "planned_end"),
}


def apply_dirt(
    tables: dict[str, pd.DataFrame], config: SynthConfig, rng: np.random.Generator
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Realistic imperfections, each counted so a test (or step 4) can check what was planted."""
    f = float(config.dirt)
    counts: dict[str, dict[str, int]] = {}
    if f <= 0:
        return tables, counts

    def note(kind: str, table: str, n: int) -> None:
        counts.setdefault(kind, {})[table] = int(n)

    for table in ("punches", "hours_daily", "absence_requests"):
        df = tables[table]
        mask = rng.random(len(df)) < f / 2
        tables[table] = pd.concat([df, df[mask]], ignore_index=True)
        note("duplicate_rows", table, mask.sum())

    df = tables["punches"]
    drop = (df["direction"].to_numpy() == "out") & (rng.random(len(df)) < f)
    tables["punches"] = df[~drop].reset_index(drop=True)
    note("missing_out", "punches", drop.sum())

    for table, columns in NULLABLE_STRINGS.items():
        df = tables[table]
        total = 0
        for column in columns:
            mask = rng.random(len(df)) < f
            df.loc[mask, column] = ""
            total += int(mask.sum())
        note("null_non_key", table, total)

    for table, (column, mapping) in BAD_ENUMS.items():
        df = tables[table]
        mask = (rng.random(len(df)) < f / 4) & df[column].isin(mapping).to_numpy()
        df.loc[mask, column] = df.loc[mask, column].map(mapping)
        note("bad_enum", table, mask.sum())

    for table, (first, second) in SWAPPED_RANGES.items():
        df = tables[table]
        mask = rng.random(len(df)) < f / 4
        a, b = df.loc[mask, first].copy(), df.loc[mask, second].copy()
        df.loc[mask, first], df.loc[mask, second] = b.to_numpy(), a.to_numpy()
        note("swapped_range", table, mask.sum())

    df = tables["punches"]
    mask = rng.random(len(df)) < f
    extra = df[mask].copy()
    shift = rng.integers(5, 41, len(extra)) * np.where(rng.random(len(extra)) < 0.5, -1, 1)
    extra["punched_at"] = extra["punched_at"] + pd.to_timedelta(shift, "min")
    extra["edited"] = True
    extra["edited_by"] = "admin"
    extra["punch_id"] = [
        "pun_" + format(i, "08d") for i in range(len(df) + 1, len(df) + len(extra) + 1)
    ]
    tables["punches"] = pd.concat([df, extra], ignore_index=True)
    note("late_edit", "punches", len(extra))

    for table in tables:
        tables[table] = _sorted(tables[table], table)
    return tables, counts


# ----------------------------------------------------------------------------- assembly


def _truth(plans: list[AccountPlan]) -> dict[str, pd.DataFrame]:
    accounts = pd.DataFrame(
        {
            "account_id": ["acc_" + format(p.index + 1, "06d") for p in plans],
            "archetype": [p.archetype for p in plans],
            "industry": [p.industry for p in plans],
            "n0": [p.n0 for p in plans],
            "has_devices": [p.has_devices for p in plans],
            "decline_start": [
                str(p.months[0] + p.decline_start) if p.decline_start is not None else ""
                for p in plans
            ],
            "trouble_start": [
                p.trouble_start.strftime("%Y-%m-%d") if p.trouble_start is not None else ""
                for p in plans
            ],
            "cancelled_at": [
                p.cancelled_at.strftime(DATETIME_FORMAT) if p.cancelled_at is not None else ""
                for p in plans
            ],
            "cancel_reason": [p.cancel_reason for p in plans],
            "forced_cancel": [p.forced_cancel for p in plans],
            "months": [len(p.months) for p in plans],
        }
    )
    rows = []
    for p in plans:
        account_id = "acc_" + format(p.index + 1, "06d")
        for m, month in enumerate(p.months):
            days = (p.cancelled_at - month.end_time).days if p.cancelled_at is not None else None
            rows.append(
                (
                    account_id,
                    str(month),
                    round(float(p.health[m]), 4),
                    round(float(p.hazard[m]), 5),
                    bool(p.scheduling_on[m]),
                    days is not None and 0 <= days < 60,
                )
            )
    monthly = pd.DataFrame(
        rows, columns=["account_id", "month", "health", "hazard", "scheduling_on", "churned_60d"]
    )
    return {"accounts_truth": accounts, "monthly_health": monthly}


def generate(
    config: SynthConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
    """Return (raw tables keyed by schema table name, truth frames, dirt counts)."""
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLE_ORDER}
    plans = []
    for index in range(config.accounts):
        rng = np.random.default_rng([config.seed, index])
        plan = plan_account(index, config, rng)
        plans.append(plan)
        for name, frame in build_account_tables(plan, config, rng).items():
            parts[name].append(frame)
    tables = _assign_ids(parts)
    tables, dirt_counts = apply_dirt(
        tables, config, np.random.default_rng([config.seed, DIRT_STREAM])
    )
    return tables, _truth(plans), dirt_counts


def format_for_csv(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Logical dtypes to the CSV conventions the raw loader reads back (empty cell = null)."""
    out = pd.DataFrame(index=df.index)
    for name, kind in columns.items():
        column = df[name]
        if kind == "datetime":
            out[name] = pd.to_datetime(column).dt.strftime(DATETIME_FORMAT).fillna("")
        elif kind == "bool":
            out[name] = column.astype(bool)
        elif kind == "int":
            out[name] = column.astype("Int64")
        elif kind == "float":
            out[name] = column.astype("float64")
        else:
            out[name] = column.fillna("").astype(str)
    return out


def write_dataset(data_dir: str | Path, config: SynthConfig, force: bool = False) -> Path:
    """Write raw/, truth/ and lookups/ under data_dir; return the raw folder."""
    data_dir = Path(data_dir)
    raw = data_dir / "raw"
    if raw.exists() and any(raw.iterdir()) and not force:
        raise FileExistsError(f"{raw} is not empty; pass --force to overwrite it")
    tables, truth, dirt_counts = generate(config)
    raw.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in TABLE_ORDER:
        spec = RAW_SCHEMA["tables"][name]
        frame = format_for_csv(tables[name], spec["columns"])
        path = write_csv_stable(frame, raw / spec["file"], float_format=CSV_FLOATS)
        files[spec["file"]] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
        }
    shutil.copyfile(SCHEMA_PATH, raw / "raw_schema.json")
    months = config.months
    manifest = {
        "source": (
            f"workforce_saas synthetic v{GENERATOR_VERSION} (seed {config.seed}, "
            f"{config.accounts} accounts, {months[0]}..{months[-1]})"
        ),
        "generator": {
            "module": __name__,
            "version": GENERATOR_VERSION,
            "seed": config.seed,
            "config": asdict(config),
            "accounts": config.accounts,
            "months": int(len(months)),
            "snapshot_date": config.snapshot_date,
        },
        "files": files,
    }
    write_json_lf(raw / "manifest.json", manifest, ndigits=None)
    truth_dir = data_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in truth.items():
        write_csv_stable(frame, truth_dir / f"{name}.csv", float_format="%.5f")
    write_json_lf(truth_dir / "dirt_counts.json", dirt_counts, ndigits=None)
    (data_dir / "lookups").mkdir(parents=True, exist_ok=True)
    return raw


def summary_line(raw: Path, data_dir: Path) -> str:
    manifest = read_json(raw / "manifest.json")
    truth = pd.read_csv(data_dir / "truth" / "monthly_health.csv")
    accounts = pd.read_csv(data_dir / "truth" / "accounts_truth.csv", keep_default_na=False)
    rows = sum(entry["rows"] for entry in manifest["files"].values())
    size = sum(entry["bytes"] for entry in manifest["files"].values()) / 1048576
    return (
        f"{raw}: {manifest['generator']['accounts']} accounts, {len(truth)} account-months, "
        f"{int((accounts['cancelled_at'] != '').sum())} cancelled, "
        f"{int(truth['churned_60d'].sum())} positive months, {rows:,} rows, {size:.1f} MB"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    defaults = SynthConfig()
    parser.add_argument(
        "--data-dir", default="data", help="folder that gets raw/, truth/ and lookups/"
    )
    parser.add_argument("--accounts", type=int, default=defaults.accounts)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--start-month", default=defaults.start_month, help="first sign-up month, YYYY-MM"
    )
    parser.add_argument(
        "--snapshot", default=defaults.snapshot_date, help="snapshot date, YYYY-MM-DD"
    )
    parser.add_argument(
        "--dirt", type=float, default=defaults.dirt, help="dirt fraction; 0 disables"
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing raw/ folder")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SynthConfig(
        accounts=args.accounts,
        seed=args.seed,
        start_month=args.start_month,
        snapshot_date=args.snapshot,
        dirt=args.dirt,
    )
    try:
        raw = write_dataset(args.data_dir, config, force=args.force)
    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(summary_line(raw, Path(args.data_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
