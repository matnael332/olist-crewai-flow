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


# Ops copilot ---------------------------------------------------------------------------------


def _copilot() -> AppTest:
    return _app("views/copilot.py")


def _quick(at: AppTest, label: str) -> AppTest:
    # AppTest serializes a single-select pills widget only when its value is set as a list.
    at.button_group[0].set_value([label]).run()
    return at


def test_copilot_page_greets_offline_without_a_key():
    at = _copilot()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "Hi! I keep an eye on the late-delivery agents." in text


def test_copilot_quick_replay_answers_instantly_with_a_table():
    at = _quick(_copilot(), "Replay 2018-07-16")
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "238 orders were purchased on 2018-07-16: 42 high" in text
    assert any("instant answer, no LLM" in c.value for c in at.caption)
    assert any(
        c.value.startswith(":material/description: Sources: clean_data.csv") for c in at.caption
    )
    assert [m.label for m in at.metric][:3] == ["High", "Elevated", "Normal"]
    assert at.dataframe


def test_copilot_chat_input_takes_text_and_csv_files():
    # AppTest cannot submit to a file-accepting chat input; free text offline is covered by
    # backend/tests/test_copilot_agent.py and the Playwright check.
    at = _copilot()
    proto = at.chat_input[0].proto
    assert list(proto.file_type) == [".csv"] and proto.max_chars == 2000


def test_copilot_passcode_gate_blocks_until_the_right_code(monkeypatch):
    monkeypatch.setenv("COPILOT_PASSCODE", "open-sesame")
    at = _copilot()
    assert not at.chat_input
    at.text_input[0].set_value("wrong").run()
    at.button[0].click().run()
    assert at.error
    at.text_input[0].set_value("open-sesame").run()
    at.button[0].click().run()
    assert at.chat_input and not at.exception


def test_copilot_message_cap_stops_the_session(monkeypatch):
    monkeypatch.setenv("COPILOT_MAX_MESSAGES", "1")
    at = _quick(_copilot(), "Which gates passed?")
    assert not at.warning
    at = _quick(at, "Replay 2018-07-16")
    assert at.warning and "used all 1 messages" in at.warning[0].value


def test_scoring_page_shows_the_backend_action():
    at = _app("views/scoring.py")
    messages = [str(m.value) for m in (*at.warning, *at.info, *at.success)]
    assert any("Monitor the shipment" in m for m in messages), messages
