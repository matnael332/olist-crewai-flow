"""Live copilot check against Gemini. Deselected by default; run with: uv run pytest -m llm"""

import os

import pytest

from pulse_copilot.agent import Copilot

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY is not set"),
]


def test_live_copilot_answers_from_the_tools():
    answer = Copilot().ask("How did the last run go?")
    assert answer.error is None, answer.text
    assert "validation_report.json" in answer.sources
    assert answer.ungrounded == []
