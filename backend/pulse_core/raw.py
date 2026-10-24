"""Raw Olist tables: schema check, typed loading and profiling.

The expected schema lives in data/raw/raw_schema.json (committed). Loading is explicit about types:
identifiers and zip prefixes stay strings (zip prefixes keep their leading zeros), timestamps are
parsed with one fixed format, and the misspelled product columns are renamed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pulse_core.io import read_json
from pulse_core.validators.report import GateReport

SCHEMA_FILE = "raw_schema.json"
RENAMES = {
    "product_name_lenght": "product_name_length",
    "product_description_lenght": "product_description_length",
}
ENCODING = "utf-8-sig"  # the translation table ships with a UTF-8 BOM


def load_raw_schema(raw_dir: str | Path) -> dict:
    return read_json(Path(raw_dir) / SCHEMA_FILE)


def read_header(path: Path) -> list[str]:
    with open(path, encoding=ENCODING, newline="") as fh:
        return [column.strip().strip('"') for column in fh.readline().rstrip("\r\n").split(",")]


def check_raw_schema(raw_dir: str | Path, gate: str = "check_inputs") -> GateReport:
    """Every raw table exists and has exactly the expected columns."""
    raw_dir = Path(raw_dir)
    report = GateReport(gate)
    try:
        schema = load_raw_schema(raw_dir)
    except (OSError, ValueError) as exc:
        report.record("raw_schema_readable", False, f"cannot read {SCHEMA_FILE}: {exc}")
        return report
    for table, spec in schema["tables"].items():
        path = raw_dir / spec["file"]
        if not path.exists():
            report.record(f"{table}_present", False, f"missing raw file {spec['file']}")
            continue
        header = read_header(path)
        expected = list(spec["columns"])
        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        report.record(
            f"{table}_columns",
            not missing and not extra,
            f"{spec['file']}: columns ok"
            if not missing and not extra
            else f"{spec['file']}: missing {missing}, unexpected {extra}",
        )
    return report


def _read_table(path: Path, columns: dict[str, str], datetime_format: str) -> pd.DataFrame:
    dtype = {name: str for name, kind in columns.items() if kind in ("string", "zip")}
    df = pd.read_csv(path, encoding=ENCODING, dtype=dtype, keep_default_na=True)
    for name, kind in columns.items():
        if kind == "datetime":
            df[name] = pd.to_datetime(df[name], format=datetime_format, errors="coerce")
        elif kind == "zip":
            df[name] = df[name].str.strip().str.zfill(5)
        elif kind == "int":
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("Int64")
        elif kind == "float":
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("float64")
    return df


def load_raw_tables(
    raw_dir: str | Path, tables: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Load raw tables with explicit types. Geolocation is skipped unless requested."""
    raw_dir = Path(raw_dir)
    schema = load_raw_schema(raw_dir)
    wanted = tables or [t for t in schema["tables"] if t != "geolocation"]
    loaded = {}
    for table in wanted:
        spec = schema["tables"][table]
        df = _read_table(raw_dir / spec["file"], spec["columns"], schema["datetime_format"])
        loaded[table] = df.rename(columns=RENAMES)
    return loaded


def profile_raw_tables(tables: dict[str, pd.DataFrame], schema: dict) -> dict:
    """Shape, nulls, duplicates and key uniqueness per raw table (raw_profile.json)."""
    profile = {}
    for table, df in tables.items():
        spec = schema["tables"][table]
        key = [RENAMES.get(c, c) for c in spec["primary_key"]]
        nulls = {col: int(n) for col, n in df.isna().sum().items() if int(n) > 0}
        entry = {
            "file": spec["file"],
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "null_counts": dict(sorted(nulls.items())),
            "exact_duplicate_rows": int(df.duplicated().sum()),
        }
        if key:
            entry["primary_key"] = key
            entry["duplicate_keys"] = int(df.duplicated(subset=key).sum())
        profile[table] = entry
    orders = tables.get("orders")
    if orders is not None:
        status = orders["order_status"].value_counts(sort=False)
        entry = profile["orders"]
        entry["status_counts"] = {str(k): int(v) for k, v in sorted(status.items())}
        purchase = orders["order_purchase_timestamp"]
        entry["purchase_min"] = purchase.min().isoformat()
        entry["purchase_max"] = purchase.max().isoformat()
    return profile
