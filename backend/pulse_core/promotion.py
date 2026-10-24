"""Promote a run's artifacts to the published artifacts/ folder by directory swap.

Copying file by file into artifacts/ could leave a half-promoted mix of two runs if Windows refuses
one replace (a file held open by Streamlit). Instead the new set is staged next to the target and
swapped in with two renames, each retried; on failure the previous folder is restored.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

RETRIES = 5
DELAY_S = 0.2


class PromotionError(RuntimeError):
    """The swap failed after retries; the previous artifacts folder was kept."""


def _rename(src: Path, dst: Path) -> None:
    for attempt in range(RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == RETRIES - 1:
                raise
            time.sleep(DELAY_S)


def promote(
    run_artifacts: str | Path, target: str | Path, run_id: str, extra: dict[str, Path] | None = None
) -> Path:
    """Replace target with a copy of run_artifacts (+ extra files at relative paths)."""
    run_artifacts, target = Path(run_artifacts), Path(target)
    staging = target.with_name(f"{target.name}.staging-{run_id}")
    previous = target.with_name(f"{target.name}.prev-{run_id}")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(run_artifacts, staging)
    for rel, src in (extra or {}).items():
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    had_target = target.exists()
    try:
        if had_target:
            _rename(target, previous)
        _rename(staging, target)
    except OSError as exc:
        if had_target and previous.exists() and not target.exists():
            _rename(previous, target)
        shutil.rmtree(staging, ignore_errors=True)
        raise PromotionError(f"could not swap {target.name}: {exc}") from exc
    if had_target:
        shutil.rmtree(previous, ignore_errors=True)
    return target
