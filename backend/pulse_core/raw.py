"""Raw tables: manifest verification, schema check, typed loading and profiling.

The expected schema lives in ``<raw_dir>/raw_schema.json`` and the pinned hashes in
``<raw_dir>/manifest.json``, both committed next to the data they describe. Loading is explicit
about types; an optional ``renames`` block in the schema fixes column names at load time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from pulse_core.columns import DATETIME_FORMAT
from pulse_core.io import read_json, sha256_file, write_csv_stable, write_json_lf
from pulse_core.validators.report import GateReport

SCHEMA_FILE = "raw_schema.json"
MANIFEST_FILE = "manifest.json"
ENCODING = "utf-8-sig"


class ManifestMismatch(RuntimeError):
    """A raw file is missing or its sha256 differs from the manifest."""


def load_raw_schema(raw_dir: str | Path) -> dict:
    return read_json(Path(raw_dir) / SCHEMA_FILE)


def verify_manifest(raw_dir: str | Path, log: Callable[[str], None] | None = None) -> dict:
    """Every file in manifest.json exists and matches its sha256. Returns the manifest."""
    raw_dir = Path(raw_dir)
    manifest = read_json(raw_dir / MANIFEST_FILE)
    problems = []
    for name, entry in sorted(manifest.get("files", {}).items()):
        path = raw_dir / name
        if not path.exists():
            problems.append(f"{name}: missing")
            continue
        digest = sha256_file(path)
        if digest != entry.get("sha256"):
            problems.append(
                f"{name}: sha256 {digest[:12]} != manifest {str(entry.get('sha256'))[:12]}"
            )
        elif log:
            log(f"{name} verified ({path.stat().st_size:,} bytes)")
    if problems:
        raise ManifestMismatch("; ".join(problems))
    return manifest


def read_header(path: Path) -> list[str]:
    with open(path, encoding=ENCODING, newline="") as fh:
        return [column.strip().strip('"') for column in fh.readline().rstrip("\r\n").split(",")]


def check_raw_schema(
    raw_dir: str | Path, gate: str = "check_inputs", required_tables: Sequence[str] = ()
) -> GateReport:
    """Every raw table exists and has exactly the expected columns, and the schema lists every
    table the spec reads (``required_tables``)."""
    raw_dir = Path(raw_dir)
    report = GateReport(gate)
    try:
        schema = load_raw_schema(raw_dir)
    except (OSError, ValueError) as exc:
        report.record("raw_schema_readable", False, f"cannot read {SCHEMA_FILE}: {exc}")
        return report
    for table in required_tables:
        if table not in schema["tables"]:
            report.record(
                f"{table}_in_schema", False, f"the spec reads {table}, not in {SCHEMA_FILE}"
            )
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


def format_for_csv(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Logical dtypes to the CSV conventions the raw loader reads back (empty cell = null)."""
    out = pd.DataFrame(index=df.index)
    for name, kind in columns.items():
        column = df[name]
        if kind == "datetime":
            out[name] = pd.to_datetime(column).dt.strftime(DATETIME_FORMAT).fillna("")
        elif kind == "bool":
            out[name] = column.map({True: "True", False: "False"}).fillna("")
        elif kind == "int":
            out[name] = column.astype("Int64")
        elif kind == "float":
            out[name] = column.astype("float64")
        else:
            out[name] = column.fillna("").astype(str)
    return out


def write_raw_dir(
    tables: dict[str, pd.DataFrame],
    schema: dict,
    raw_dir: str | Path,
    manifest_extra: dict,
    float_format: str,
) -> Path:
    """Write the raw layer the pipeline reads: one CSV per table of ``schema`` (in schema
    order), ``raw_schema.json`` (``schema`` itself) and ``manifest.json`` (``manifest_extra``
    plus the bytes, sha256 and rows of every file). Returns ``raw_dir``."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, spec in schema["tables"].items():
        frame = format_for_csv(tables[name], spec["columns"])
        path = write_csv_stable(frame, raw_dir / spec["file"], float_format=float_format)
        files[spec["file"]] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
        }
    write_json_lf(raw_dir / SCHEMA_FILE, schema, ndigits=None)
    write_json_lf(raw_dir / MANIFEST_FILE, {**manifest_extra, "files": files}, ndigits=None)
    return raw_dir


def _read_table(path: Path, columns: dict[str, str], datetime_format: str) -> pd.DataFrame:
    dtype = {name: str for name, kind in columns.items() if kind == "string"}
    df = pd.read_csv(path, encoding=ENCODING, dtype=dtype, keep_default_na=True)
    for name, kind in columns.items():
        if kind == "datetime":
            df[name] = pd.to_datetime(df[name], format=datetime_format, errors="coerce")
        elif kind == "int":
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("Int64")
        elif kind == "float":
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("float64")
        elif kind == "bool":
            df[name] = df[name].map({"True": True, "False": False, True: True, False: False})
    return df


def load_raw_tables(
    raw_dir: str | Path, tables: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Load the raw tables named in the schema with explicit types."""
    raw_dir = Path(raw_dir)
    schema = load_raw_schema(raw_dir)
    renames = schema.get("renames", {})
    loaded = {}
    for table in tables or list(schema["tables"]):
        spec = schema["tables"][table]
        df = _read_table(raw_dir / spec["file"], spec["columns"], schema["datetime_format"])
        loaded[table] = df.rename(columns=renames)
    return loaded


def profile_raw_tables(tables: dict[str, pd.DataFrame], schema: dict) -> dict:
    """Shape, nulls, duplicates and key uniqueness per raw table (raw_profile.json)."""
    renames = schema.get("renames", {})
    profile = {}
    for table, df in tables.items():
        spec = schema["tables"][table]
        key = [renames.get(c, c) for c in spec.get("primary_key", [])]
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
    return profile
