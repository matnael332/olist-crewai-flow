"""Streamlit dashboard smoke tests with AppTest (every page renders without exceptions)."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[1] / "app"
PAGES = [
    "views/overview.py",
    "views/data_contract.py",
    "views/eda_report.py",
    "views/model.py",
    "views/predict.py",
    "views/run_logs.py",
]


@pytest.fixture(scope="module", autouse=True)
def artifacts_env(pipeline_out):
    patch = pytest.MonkeyPatch()
    patch.setenv("ARTIFACTS_DIR", str(pipeline_out))
    patch.syspath_prepend(str(APP))
    sys.modules.pop("data_access", None)
    yield
    patch.undo()
    sys.modules.pop("data_access", None)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=120)
    at.run()
    if page != PAGES[0]:
        at.switch_page(page)
        at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_predict_scores_the_default_order():
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=120)
    at.run()
    at.switch_page("views/predict.py")
    at.run()
    next(b for b in at.button if b.label == "Score order").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Late-delivery probability" in [m.label for m in at.metric]
    messages = [str(m.value) for m in (*at.warning, *at.info, *at.success)]
    assert sum("risk" in m for m in messages) == 1, messages


def test_overview_shows_status_and_metrics():
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=120)
    at.run()
    # The fixture artifacts were produced without the Flow (no run_status/run_manifest), so the
    # overview must say it cannot trust them instead of claiming success.
    assert at.warning and not at.success
    labels = [m.label for m in at.metric]
    assert "Delivered orders" in labels and "Model ROC-AUC (test)" in labels
