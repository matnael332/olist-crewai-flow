"""run_status.json: the single source of truth for "where is this run and did it pass".

Rewritten atomically (with retries) at every stage transition; the dashboard reads it instead of
parsing logs, and tolerates a file that is momentarily locked or missing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from olist_core.io import read_json, write_json_lf

STATUS_FILE = "run_status.json"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunStatus:
    def __init__(self, path: str | Path, **initial) -> None:
        self.path = Path(path)
        self.data = {
            "status": "running",
            "stage": "init",
            "stages_done": [],
            "errors": [],
            "warnings": [],
            "started_at": utcnow(),
            **initial,
        }
        self.write()

    def write(self) -> None:
        write_json_lf(self.path, self.data, ndigits=None)

    def stage_start(self, stage: str) -> None:
        self.data.update(stage=stage, stage_started_at=utcnow())
        self.write()

    def stage_done(self, stage: str) -> None:
        if stage not in self.data["stages_done"]:
            self.data["stages_done"].append(stage)
        self.write()

    def update(self, **fields) -> None:
        self.data.update(fields)
        self.write()


def read_status(path: str | Path) -> dict | None:
    """Tolerant reader: None when the file is missing, locked or half-written."""
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None
