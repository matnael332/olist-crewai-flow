"""Streamlit dashboard tests with AppTest, against the committed artifacts."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ENTRY = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
PAGES = [
    "views/overview.py",
    "views/insights.py",
    "views/scoring.py",
    "views/model.py",
    "views/data_contract.py",
    "views/pipeline_runs.py",
]
RISK_LABELS = ("High risk", "Elevated risk", "Normal risk")


def _app(page: str | None = None) -> AppTest:
    at = AppTest.from_file(ENTRY, default_timeout=180)
    at.run()
    if page and page != PAGES[0]:
        at.switch_page(page)
        at.run()
    return at


def _risk_labels(at: AppTest) -> list[str]:
    messages = [str(m.value) for m in (*at.warning, *at.info, *at.success)]
    return [label for message in messages for label in RISK_LABELS if f"**{label}.**" in message]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = _app(page)
    assert not at.exception, [e.value for e in at.exception]


def test_overview_trusts_the_promoted_run():
    at = _app()
    assert not at.warning
    labels = [m.label for m in at.metric]
    assert "Orders delivered late" in labels
    assert "Test ROC-AUC" in labels


def test_overview_warns_without_a_promoted_run(untrusted_artifacts):
    at = _app()
    assert at.warning


def test_scoring_long_distance_preset_is_elevated_risk():
    at = _app("views/scoring.py")
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Late-delivery probability" in labels
    assert {"Seller in the customer's state", "Promise 7 more days", "Pay by credit card"} <= set(
        labels
    )
    assert _risk_labels(at) == ["Elevated risk"]


def test_scoring_same_state_order_is_normal_risk():
    at = _app("views/scoring.py")
    # AppTest serializes a single-select segmented control only when its value is set as a list.
    at.button_group(key="preset").set_value(["Long distance: SP to BA"])
    at.selectbox(key="cs-long-distance").set_value("SP")
    next(b for b in at.button if b.label == "Score order").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert _risk_labels(at) == ["Normal risk"]
