import logging

from olist_core.logging_setup import Redactor, RunLog, get_logger

FAKE_KEY = "AIza" + "B" * 35  # built at runtime so the repo never contains a key-shaped literal


def test_redactor_masks_env_secrets_and_key_shapes(monkeypatch):
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "super-secret-token-value")
    redact = Redactor()
    text = redact(f"token=super-secret-token-value key={FAKE_KEY}")
    assert "super-secret-token-value" not in text
    assert FAKE_KEY not in text
    assert text.count("***") == 2


def test_runlog_writes_redacted_file_and_detaches(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    log_file = tmp_path / "pipeline.log"
    with RunLog(log_file, console=False):
        get_logger("test").info("using key %s", FAKE_KEY)
        try:
            raise RuntimeError(f"boom {FAKE_KEY}")
        except RuntimeError:
            get_logger("test").exception("failure")
    content = log_file.read_text(encoding="utf-8")
    assert FAKE_KEY not in content
    assert "using key ***" in content and "boom ***" in content
    assert logging.getLogger("olist").handlers == []
