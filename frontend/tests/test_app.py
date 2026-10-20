"""Streamlit dashboard tests with AppTest, against the committed artifacts."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ENTRY = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
PAGES = [
    "views/overview.py",
    "views/data_contract.py",
    "views/eda_report.py",
    "views/model.py",
    "views/predict.py",
    "views/run_logs.py",
]


def _app(page: str | None = None) -> AppTest:
    at = AppTest.from_file(ENTRY, default_timeout=120)
    at.run()
    if page and page != PAGES[0]:
        at.switch_page(page)
        at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = _app(page)
    assert not at.exception, [e.value for e in at.exception]


def test_overview_trusts_the_promoted_run():
    at = _app()
    assert at.success and not at.warning
    labels = [m.label for m in at.metric]
    assert "Delivered orders" in labels and "Model ROC-AUC (test)" in labels


def test_overview_warns_without_a_promoted_run(untrusted_artifacts):
    at = _app()
    assert at.warning and not at.success


def test_predict_scores_the_default_order():
    at = _app("views/predict.py")
    next(b for b in at.button if b.label == "Score order").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Late-delivery probability" in [m.label for m in at.metric]
    messages = [str(m.value) for m in (*at.warning, *at.info, *at.success)]
    assert sum("risk" in m for m in messages) == 1, messages
