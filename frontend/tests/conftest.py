"""Frontend test setup: the dashboard is tested against the committed artifacts/ it deploys with."""

import shutil
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
    monkeypatch.delenv("API_URL", raising=False)
    # The copilot must be deterministic in tests: no Gemini key, no passcode, default cap.
    for name in ("GEMINI_API_KEY", "COPILOT_PASSCODE", "COPILOT_MAX_MESSAGES", "COPILOT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    _forget_frontend_modules()
    yield
    _forget_frontend_modules()


@pytest.fixture
def untrusted_artifacts(tmp_path, monkeypatch):
    """A copy of the small artifact files without run_status/run_manifest (no promoted run)."""
    for path in (ROOT / "artifacts").iterdir():
        if path.suffix in {".json", ".md"} and path.name not in {
            "run_status.json",
            "run_manifest.json",
        }:
            shutil.copy2(path, tmp_path / path.name)
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    return tmp_path
