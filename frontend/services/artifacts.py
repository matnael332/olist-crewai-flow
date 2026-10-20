"""Cached, tolerant loaders for the dashboard. Reads artifacts and run status files only; never
imports CrewAI. Caches are keyed by file modification time so new runs show up immediately."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401

ROOT = _bootstrap.ROOT
ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
RUNS = ROOT / "runs"


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _read_text(path: str, mtime: float) -> str:
    return Path(path).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float, nrows: int | None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, dtype={"order_id": str, "customer_zip_prefix": str})


def load_json(name: str, base: Path = ARTIFACTS):
    path = base / name
    try:
        return _read_json(str(path), _mtime(path)) if path.exists() else None
    except (OSError, ValueError):
        return None


def load_text(name: str, base: Path = ARTIFACTS) -> str | None:
    path = base / name
    try:
        return _read_text(str(path), _mtime(path)) if path.exists() else None
    except OSError:
        return None


def load_csv(name: str, nrows: int | None = None, base: Path = ARTIFACTS) -> pd.DataFrame | None:
    path = base / name
    try:
        return _read_csv(str(path), _mtime(path), nrows) if path.exists() else None
    except (OSError, ValueError):
        return None


def figure(name: str) -> Path | None:
    path = ARTIFACTS / "figures" / name
    return path if path.exists() else None


def artifacts_valid() -> tuple[bool, str]:
    """artifacts/ is trusted only when its status and manifest describe the same run."""
    status, manifest = load_json("run_status.json"), load_json("run_manifest.json")
    if not status or not manifest:
        return False, "no promoted run found in artifacts/"
    if manifest.get("run", {}).get("run_id") != status.get("run_id"):
        return False, "artifacts/ status and manifest belong to different runs"
    return True, status.get("run_id", "")


def list_runs() -> list[dict]:
    """Promoted run plus local runs (runs/*/run_status.json), newest first."""
    runs = []
    promoted = load_json("run_status.json")
    if promoted:
        runs.append({**promoted, "_source": "promoted (artifacts/)", "_dir": str(ARTIFACTS)})
    if RUNS.exists():
        for status_file in RUNS.glob("*/run_status.json"):
            status = load_json("run_status.json", status_file.parent)
            if status:
                runs.append({**status, "_source": "local run", "_dir": str(status_file.parent)})
    return sorted(runs, key=lambda r: r.get("started_at") or "", reverse=True)


def pct(value, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"
