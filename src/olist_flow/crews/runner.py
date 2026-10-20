"""Crew kickoff indirection.

The Flow never kicks off a crew directly; it calls narrate(phase, ...). Tests monkeypatch this
function to simulate LLM outcomes (structured output missing, quota exhausted) without network.
The real crews are wired in milestone M4.
"""

from __future__ import annotations

from pathlib import Path


def narrate(phase: str, artifacts_dir: Path, context: dict) -> dict[str, str]:
    """Run the crew for a phase and re-render narrative artifacts. Returns narrative sources."""
    raise NotImplementedError("LLM crews are added in milestone M4")
