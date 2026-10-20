"""Deterministic, Windows-safe file helpers.

Every artifact is written through these helpers so that:
- text is UTF-8 with LF line endings on every OS (byte-identical on Windows and Linux CI),
- JSON is key-sorted, indented and free of numpy/pandas scalar reprs,
- files are replaced atomically, retrying while another process (e.g. Streamlit) holds them open.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CSV_FLOAT_FORMAT = "%.4f"
CSV_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
REPLACE_RETRIES = 5
REPLACE_DELAY_S = 0.2


def atomic_replace(
    src: str | os.PathLike, dst: str | os.PathLike, retries: int = REPLACE_RETRIES
) -> None:
    """os.replace with retries: Windows refuses to replace a file another process has open."""
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(REPLACE_DELAY_S)


def write_text_lf(path: str | os.PathLike, text: str) -> Path:
    """Write text as UTF-8 with LF newlines via a temp file and an atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    atomic_replace(tmp, path)
    return path


def read_text(path: str | os.PathLike) -> str:
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def to_builtin(obj: Any) -> Any:
    """Recursively convert numpy/pandas/path objects into JSON-safe builtins.

    NaN and infinities become None (strict JSON has no NaN); dict keys become strings.
    """
    if obj is None or isinstance(obj, str | bool):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, os.PathLike):
        return Path(obj).as_posix()
    if isinstance(obj, Mapping):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset | np.ndarray | pd.Series | pd.Index):
        items = list(obj)
        if isinstance(obj, set | frozenset):
            items = sorted(items, key=str)
        return [to_builtin(v) for v in items]
    if obj is pd.NA or obj is pd.NaT:
        return None
    raise TypeError(f"Cannot convert {type(obj).__name__} to a JSON builtin")


def round_floats(obj: Any, ndigits: int = 6) -> Any:
    """Round every float inside a nested structure (stabilizes JSON across machines)."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def dumps_json(data: Any, ndigits: int | None = 6) -> str:
    payload = to_builtin(data)
    if ndigits is not None:
        payload = round_floats(payload, ndigits)
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_json_lf(path: str | os.PathLike, data: Any, ndigits: int | None = 6) -> Path:
    return write_text_lf(path, dumps_json(data, ndigits))


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_csv_stable(df: pd.DataFrame, path: str | os.PathLike) -> Path:
    """Write a CSV that is byte-identical across runs and operating systems."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(
        tmp,
        index=False,
        lineterminator="\n",
        float_format=CSV_FLOAT_FORMAT,
        date_format=CSV_DATE_FORMAT,
        encoding="utf-8",
    )
    atomic_replace(tmp, path)
    return path


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
