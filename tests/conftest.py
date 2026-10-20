"""Shared pytest configuration and fixtures."""

import json
import os
from pathlib import Path

import pytest

# Keep CrewAI quiet and offline in tests, before anything imports it.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_TESTING", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "olist_mini"


@pytest.fixture(scope="session")
def fixture_raw_dir() -> Path:
    return FIXTURE_DIR / "raw"


@pytest.fixture(scope="session")
def fixture_lookups_dir() -> Path:
    return FIXTURE_DIR / "lookups"


@pytest.fixture(scope="session")
def injections() -> dict:
    return json.loads((FIXTURE_DIR / "injections.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def analyst_out(tmp_path_factory, fixture_raw_dir, fixture_lookups_dir) -> Path:
    """One deterministic analyst-stage run on the fixture, shared by read-only tests."""
    from olist_core.stages import run_analyst_stage

    out = tmp_path_factory.mktemp("analyst")
    run_analyst_stage(fixture_raw_dir, fixture_lookups_dir, out)
    return out
