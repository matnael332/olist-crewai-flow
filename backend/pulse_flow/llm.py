"""Gemini LLM factory for CrewAI 1.14.5 (native google-genai provider).

Verified against the installed provider:
- the model string needs the "gemini/" prefix (otherwise it routes to OpenAI);
- seed / max_tokens / timeout / max_retries are silently dropped, so the timeout and retries go
  through google-genai HttpOptions and the output cap through max_output_tokens;
- temperature is ignored on Gemini 3.6+, so LLM text is never bit-reproducible (reproducibility
  comes from the deterministic Python stages, not from the model);
- a ThinkingConfig is always passed: MINIMAL for flash-lite, LOW for 3.8-flash.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def resolve_model(model: str | None = None) -> str:
    name = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
    return name.removeprefix("gemini/")


def has_gemini_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def make_llm(model: str | None = None):
    """Create one LLM instance (use one per agent per run: token usage is tracked per instance)."""
    from crewai import LLM
    from google.genai import types

    name = resolve_model(model)
    lite = "flash-lite" in name
    thinking_level = types.ThinkingLevel.MINIMAL if lite else types.ThinkingLevel.LOW
    retry = types.HttpRetryOptions(
        attempts=5,
        initial_delay=10.0,
        max_delay=60.0,
        exp_base=2.0,
        jitter=2.0,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    )
    return LLM(
        model=f"gemini/{name}",
        max_output_tokens=4096 if lite else 8192,
        stream=False,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level, include_thoughts=False),
        client_params={"http_options": types.HttpOptions(timeout=120_000, retry_options=retry)},
    )
