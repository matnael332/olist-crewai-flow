"""Frontend test setup: isolate the environment so the pages read ARTIFACTS_DIR afresh."""

import sys
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1]
ROOT = FRONTEND.parent


def _forget_frontend_modules() -> None:
    """Drop cached frontend modules so each test reads ARTIFACTS_DIR afresh."""
    for name in [m for m in sys.modules if m == "services" or m.startswith("services.")]:
        sys.modules.pop(name)


@pytest.fixture(autouse=True)
def frontend_env(monkeypatch):
    monkeypatch.syspath_prepend(str(FRONTEND))
    monkeypatch.delenv("ARTIFACTS_DIR", raising=False)
    # The copilot must be deterministic in tests: no Gemini key, no passcode, default cap.
    for name in ("GEMINI_API_KEY", "COPILOT_PASSCODE", "COPILOT_MAX_MESSAGES", "COPILOT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    _forget_frontend_modules()
    yield
    _forget_frontend_modules()
