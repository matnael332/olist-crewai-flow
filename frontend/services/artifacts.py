"""Cached, tolerant loaders for the dashboard.

Reads the committed artifacts and run status files only; never imports CrewAI. Caches are keyed by
file modification time, so a newly promoted run shows up without restarting the app. Missing or
unreadable files return None and the pages show an empty state instead of an error.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import _bootstrap

ROOT = _bootstrap.ROOT
ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
RUNS = ROOT / "runs"

STAGE_END = re.compile(r"STAGE (\w+) END status=(\w+) elapsed=([\d.]+)s")
LOG_LINE = re.compile(r"^\d\d:\d\d:\d\d\s+(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")


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
def _read_csv(
    path: str, mtime: float, nrows: int | None, usecols: tuple[str, ...] | None
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        nrows=nrows,
        usecols=list(usecols) if usecols else None,
        dtype={"order_id": str, "customer_zip_prefix": str},
    )


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


def load_csv(
    name: str,
    nrows: int | None = None,
    usecols: tuple[str, ...] | None = None,
    base: Path = ARTIFACTS,
) -> pd.DataFrame | None:
    path = base / name
    try:
        return _read_csv(str(path), _mtime(path), nrows, usecols) if path.exists() else None
    except (OSError, ValueError):
        return None


# Typed accessors for the files every page needs ----------------------------------------------


def eda() -> dict | None:
    return load_json("eda_summary.json")


def metrics() -> dict | None:
    return load_json("metrics.json")


def model_meta() -> dict | None:
    return load_json("model_meta.json")


def contract() -> dict | None:
    return load_json("dataset_contract.json")


def cleaning() -> dict | None:
    return load_json("cleaning_report.json")


def status() -> dict | None:
    return load_json("run_status.json")


def manifest() -> dict | None:
    return load_json("run_manifest.json")


def validation(base: Path = ARTIFACTS) -> dict | None:
    return load_json("validation_report.json", base)


def artifacts_valid() -> tuple[bool, str]:
    """artifacts/ is trusted only when its status and manifest describe the same run."""
    run_status, run_manifest = status(), manifest()
    if not run_status or not run_manifest:
        return False, "no promoted run found in artifacts/"
    if run_manifest.get("run", {}).get("run_id") != run_status.get("run_id"):
        return False, "artifacts/ status and manifest belong to different runs"
    return True, run_status.get("run_id", "")


def gate_summary(report: dict | None) -> dict[str, dict]:
    """Gate name -> {'passed', 'n_passed', 'n_total', 'checks'}."""
    if not report:
        return {}
    out = {}
    for gate, result in report.get("gates", {}).items():
        checks = result.get("checks", [])
        out[gate] = {
            "passed": bool(result.get("passed")),
            "n_passed": sum(1 for c in checks if c.get("passed")),
            "n_total": len(checks),
            "checks": checks,
        }
    return out


def insights_headline() -> str | None:
    """The 'Headline' paragraph of insights.md (written by the Data Analyst crew)."""
    text = load_text("insights.md")
    if not text or "## Headline" not in text:
        return None
    body = text.split("## Headline", 1)[1].split("\n## ", 1)[0].strip()
    return body or None


def review_shares() -> pd.DataFrame | None:
    """Share of each review score among late and on-time delivered orders."""
    df = load_csv("clean_data.csv", usecols=("review_score", "is_late"))
    if df is None:
        return None
    df = df.dropna(subset=["review_score"])
    df["delivery"] = np.where(df["is_late"].astype(int) == 1, "Late", "On time")
    df["score"] = df["review_score"].astype(int)
    counts = df.groupby(["delivery", "score"]).size().rename("n").reset_index()
    counts["share"] = counts["n"] / counts.groupby("delivery")["n"].transform("sum")
    return counts


# Model evaluation curves, computed from the committed test-window predictions -----------------


def _thin(n: int, keep: int = 300) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(n, keep)).astype(int))


@st.cache_data(show_spinner="Computing evaluation curves...")
def _curves(path: str, mtime: float, models: tuple[str, ...], selected: str):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    df = pd.read_csv(path)
    y = df["y_true"].to_numpy()
    roc_rows, pr_rows, calibration_rows = [], [], []
    for name in models:
        column = f"p_{name}"
        if column not in df:
            continue
        p = df[column].to_numpy()
        if np.unique(p).size == 1:  # constant baseline: the chance diagonal, no PR curve
            roc_rows += [
                {"model": name, "fpr": 0.0, "tpr": 0.0},
                {"model": name, "fpr": 1.0, "tpr": 1.0},
            ]
            continue
        fpr, tpr, _ = roc_curve(y, p)
        roc_rows += [{"model": name, "fpr": fpr[i], "tpr": tpr[i]} for i in _thin(len(fpr))]
        precision, recall, _ = precision_recall_curve(y, p)
        precision, recall = precision[:-1], recall[:-1]  # drop the threshold-less end point
        keep = recall > 0  # points before the first late order is caught carry no information
        precision, recall = precision[keep], recall[keep]
        pr_rows += [
            {"model": name, "recall": recall[i], "precision": precision[i]}
            for i in _thin(len(recall))
        ]
        if name == selected:
            observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
            calibration_rows = [
                {"mean_predicted": m, "observed": o}
                for m, o in zip(predicted, observed, strict=True)
            ]
    return (
        pd.DataFrame(roc_rows),
        pd.DataFrame(pr_rows),
        pd.DataFrame(calibration_rows),
        float(y.mean()),
    )


def model_curves(models: tuple[str, ...], selected: str):
    """(roc, pr, calibration, prevalence) for the test window, or None without predictions."""
    path = ARTIFACTS / "predictions_test.csv"
    if not path.exists():
        return None
    return _curves(str(path), _mtime(path), models, selected)


# Runs and logs -----------------------------------------------------------------------------------


def list_runs() -> list[dict]:
    """Promoted run plus local runs (runs/*/run_status.json), newest first."""
    runs = []
    promoted = status()
    if promoted:
        runs.append({**promoted, "_source": "promoted (artifacts/)", "_dir": str(ARTIFACTS)})
    if RUNS.exists():
        for status_file in RUNS.glob("*/run_status.json"):
            run_status = load_json("run_status.json", status_file.parent)
            if run_status:
                runs.append({**run_status, "_source": "local run", "_dir": str(status_file.parent)})
    return sorted(runs, key=lambda r: r.get("started_at") or "", reverse=True)


def run_log(folder: Path) -> str | None:
    return load_text("logs/pipeline.log", folder) or load_text("pipeline.log", folder)


def stage_durations(log: str | None) -> list[dict]:
    """[{'stage', 'status', 'seconds'}] from the 'STAGE <name> END status= elapsed=' log lines."""
    if not log:
        return []
    return [
        {"stage": m.group(1), "status": m.group(2), "seconds": float(m.group(3))}
        for m in STAGE_END.finditer(log)
    ]


def log_level(line: str) -> str | None:
    match = LOG_LINE.match(line)
    return match.group(1) if match else None
