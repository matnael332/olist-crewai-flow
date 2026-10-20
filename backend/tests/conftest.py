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


@pytest.fixture(scope="session")
def pipeline_out(tmp_path_factory, analyst_out) -> Path:
    """Analyst output plus the feature and modeling stages (deterministic), shared read-only."""
    import shutil

    from olist_core.stages import run_feature_stage, run_modeling_stage

    out = tmp_path_factory.mktemp("pipeline") / "artifacts"
    shutil.copytree(analyst_out, out)
    run_feature_stage(out)
    run_modeling_stage(out)
    return out


@pytest.fixture()
def pipeline_copy(tmp_path, pipeline_out) -> Path:
    """A private, mutable copy of the full pipeline output."""
    import shutil

    target = tmp_path / "artifacts"
    shutil.copytree(pipeline_out, target)
    return target


@pytest.fixture(scope="session")
def bundle(pipeline_out, fixture_lookups_dir):
    """The fixture model loaded the way the API loads it."""
    from olist_api.model import ModelBundle

    loaded = ModelBundle(pipeline_out, fixture_lookups_dir)
    assert loaded.ok, loaded.error
    return loaded


@pytest.fixture(scope="session")
def api_record(pipeline_out) -> dict:
    """A valid /predict record for the fixture model (its category comes from the fixture)."""
    from olist_core.columns import read_clean_data

    clean = read_clean_data(pipeline_out / "clean_data.csv")
    return {
        "n_items": 1,
        "total_price": 129.9,
        "total_freight": 21.5,
        "customer_state": "BA",
        "seller_state": "SP",
        "primary_category": clean["primary_category"].iloc[0],
        "payment_type": "boleto",
        "max_installments": 1,
        "purchase_ts": "2018-06-12 14:03:00",
        "estimated_delivery_date": "2018-07-05",
    }


@pytest.fixture(scope="session")
def busiest_test_day(pipeline_out):
    """The test-window purchase date with the most fixture orders (a realistic replay day)."""
    from olist_core.columns import read_clean_data

    clean = read_clean_data(pipeline_out / "clean_data.csv")
    days = clean["purchase_ts"].dt.date
    in_window = days[(clean["purchase_ts"] >= "2018-05-01") & (clean["purchase_ts"] < "2018-08-01")]
    return in_window.value_counts().sort_index().idxmax()
