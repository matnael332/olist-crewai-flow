"""Repository root discovery, so no module depends on how deep it sits in the tree."""

from __future__ import annotations

import os
from pathlib import Path

MARKER = "pyproject.toml"


def repo_root() -> Path:
    """Return the folder that holds pyproject.toml, found by walking up from this file.

    The PULSE_REPO_ROOT environment variable overrides the search (e.g. for an installed wheel).
    """
    override = os.getenv("PULSE_REPO_ROOT")
    if override:
        return Path(override).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / MARKER).is_file():
            return parent
    return Path.cwd().resolve()
