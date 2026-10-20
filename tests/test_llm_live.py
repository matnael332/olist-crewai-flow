"""Live Gemini checks. Deselected by default; run with: uv run pytest -m llm"""

import pytest

from olist_flow.llm import has_gemini_key, make_llm

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not has_gemini_key(), reason="GEMINI_API_KEY is not set"),
]


def test_gemini_native_provider_round_trip():
    llm = make_llm()
    assert type(llm).__name__ == "GeminiCompletion"
    assert llm.supports_function_calling()
    reply = llm.call("Reply with the single word OK and nothing else.")
    assert "OK" in str(reply).upper()
