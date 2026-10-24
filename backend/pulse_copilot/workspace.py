"""Read-only access to the committed artifacts and local runs (no Streamlit, no LLM).

Also holds the small pure helpers the dashboard and the copilot share: gate summaries, stage
timings from pipeline.log and the insights headline.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from pulse_api.model import ModelBundle
from pulse_api.replay import load_clean
from pulse_core.paths import repo_root

STAGE_END = re.compile(r"STAGE (\w+) END status=(\w+) elapsed=([\d.]+)s")


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@lru_cache(maxsize=128)
def _read_text(path: str, mtime: float) -> str:
    return Path(path).read_text(encoding="utf-8")


@lru_cache(maxsize=2)
def _bundle(artifacts: str, lookups: str, mtime: float) -> ModelBundle:
    return ModelBundle(Path(artifacts), Path(lookups))


def gate_summary(report: dict | None) -> dict[str, dict]:
    """Gate name -> {'passed', 'n_passed', 'n_total', 'checks'}."""
    out = {}
    for gate, result in (report or {}).get("gates", {}).items():
        checks = result.get("checks", [])
        out[gate] = {
            "passed": bool(result.get("passed")),
            "n_passed": sum(1 for c in checks if c.get("passed")),
            "n_total": len(checks),
            "checks": checks,
        }
    return out


def stage_durations(log: str | None) -> list[dict]:
    """[{'stage', 'status', 'seconds'}] from the 'STAGE <name> END status= elapsed=' lines."""
    if not log:
        return []
    return [
        {"stage": m.group(1), "status": m.group(2), "seconds": float(m.group(3))}
        for m in STAGE_END.finditer(log)
    ]


def insights_headline(markdown: str | None) -> str | None:
    """The 'Headline' paragraph of insights.md (written by the Data Analyst crew)."""
    if not markdown or "## Headline" not in markdown:
        return None
    body = markdown.split("## Headline", 1)[1].split("\n## ", 1)[0].strip()
    return body or None


def duration_seconds(started: str | None, finished: str | None) -> float | None:
    try:
        return (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Workspace:
    root: Path
    artifacts: Path
    lookups: Path
    runs: Path

    @classmethod
    def from_env(cls, artifacts_dir: str | Path | None = None) -> Workspace:
        root = repo_root()
        artifacts = Path(artifacts_dir or os.getenv("ARTIFACTS_DIR", root / "artifacts"))
        return cls(root, artifacts, root / "data" / "lookups", root / "runs")

    def text(self, name: str, base: Path | None = None) -> str | None:
        path = (base or self.artifacts) / name
        try:
            return _read_text(str(path), _mtime(path)) if path.exists() else None
        except OSError:
            return None

    def json(self, name: str, base: Path | None = None) -> dict | None:
        text = self.text(name, base)
        try:
            return json.loads(text) if text else None
        except ValueError:
            return None

    def bundle(self) -> ModelBundle:
        model = self.artifacts / "model.joblib"
        return _bundle(str(self.artifacts), str(self.lookups), _mtime(model))

    def clean(self) -> pd.DataFrame:
        return load_clean(self.artifacts)

    def promoted_run_id(self) -> str | None:
        return (self.json("run_status.json") or {}).get("run_id")

    def run_folder(self, run_id: str = "") -> Path | None:
        """The promoted run (artifacts/) when run_id is empty or matches it, else runs/<id>."""
        if not run_id or run_id == self.promoted_run_id():
            return self.artifacts
        folder = self.runs / run_id
        return folder if (folder / "run_status.json").exists() else None

    @staticmethod
    def report_dir(folder: Path) -> Path:
        return folder if (folder / "validation_report.json").exists() else folder / "artifacts"

    def run_log(self, folder: Path) -> str | None:
        return self.text("logs/pipeline.log", folder) or self.text("pipeline.log", folder)

    def list_runs(self) -> list[dict]:
        """The promoted run plus local runs (runs/*/run_status.json), newest first."""
        runs = []
        promoted = self.json("run_status.json")
        if promoted:
            runs.append(
                {**promoted, "_source": "promoted (artifacts/)", "_dir": str(self.artifacts)}
            )
        if self.runs.exists():
            for status_file in self.runs.glob("*/run_status.json"):
                status = self.json("run_status.json", status_file.parent)
                if status:
                    runs.append({**status, "_source": "local run", "_dir": str(status_file.parent)})
        return sorted(runs, key=lambda r: r.get("started_at") or "", reverse=True)
