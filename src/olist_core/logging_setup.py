"""Run logging: a dedicated "olist" logger with secret redaction.

- Never configures the root logger (keeps pytest's and Streamlit's handlers intact).
- Every handler formats through a redactor that masks the values of *_KEY / *_TOKEN / *_SECRET
  environment variables and anything that looks like a Google API key, including tracebacks.
- Third-party loggers are quieted to WARNING, except httpx at INFO (one line per Gemini HTTP call).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

LOGGER_NAME = "olist"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
DATE_FORMAT = "%H:%M:%S"

_SECRET_ENV = re.compile(r"(_KEY|_TOKEN|_SECRET|_PASSWORD)$", re.IGNORECASE)
_GOOGLE_KEY = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
_QUIET_LOGGERS = (
    "crewai",
    "httpcore",
    "urllib3",
    "google",
    "google_genai",
    "opentelemetry",
    "chromadb",
    "matplotlib",
    "PIL",
    "lancedb",
)


class Redactor:
    """Masks secret values. Secrets are read from the environment at construction time."""

    def __init__(self, extra_secrets: list[str] | None = None) -> None:
        values = {
            value
            for key, value in os.environ.items()
            if _SECRET_ENV.search(key) and value and len(value) >= 8
        }
        values.update(s for s in (extra_secrets or []) if s and len(s) >= 8)
        self._secrets = sorted(values, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return _GOOGLE_KEY.sub("***", text)


class RedactingFormatter(logging.Formatter):
    def __init__(self, redactor: Redactor) -> None:
        super().__init__(LOG_FORMAT, DATE_FORMAT)
        self._redact = redactor

    def format(self, record: logging.LogRecord) -> str:
        return self._redact(super().format(record))


class LFFileHandler(logging.FileHandler):
    """File handler that writes LF line endings on every OS (the log is a committed artifact)."""

    def _open(self):
        return open(  # noqa: SIM115 - the handler owns and closes the stream
            self.baseFilename, self.mode, encoding=self.encoding, errors=self.errors, newline="\n"
        )


class SafeStreamHandler(logging.StreamHandler):
    """Console handler that never crashes on characters the console code page cannot encode."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            encoding = getattr(self.stream, "encoding", None) or "utf-8"
            message = message.encode(encoding, errors="backslashreplace").decode(
                encoding, errors="replace"
            )
            self.stream.write(message + self.terminator)
            self.flush()
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


class RunLog:
    """Attach file + console handlers to the "olist" logger for the duration of a run."""

    def __init__(self, log_file: str | os.PathLike | None, console: bool = True) -> None:
        self.log_file = Path(log_file) if log_file else None
        self.console = console
        self._handlers: list[logging.Handler] = []
        self._saved_levels: dict[str, int] = {}

    def __enter__(self) -> RunLog:
        formatter = RedactingFormatter(Redactor())
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = LFFileHandler(self.log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self._handlers.append(file_handler)
        if self.console:
            console_handler = SafeStreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self._handlers.append(console_handler)

        for name, level, attach in [
            (LOGGER_NAME, logging.INFO, True),
            ("httpx", logging.INFO, True),
        ]:
            logger = logging.getLogger(name)
            self._saved_levels[name] = logger.level
            logger.setLevel(level)
            logger.propagate = False
            if attach:
                for handler in self._handlers:
                    logger.addHandler(handler)
        for name in _QUIET_LOGGERS:
            logger = logging.getLogger(name)
            self._saved_levels[name] = logger.level
            logger.setLevel(logging.WARNING)
        return self

    def close(self) -> None:
        """Flush and detach handlers (call before copying the log file elsewhere)."""
        for name in (LOGGER_NAME, "httpx"):
            logger = logging.getLogger(name)
            for handler in self._handlers:
                logger.removeHandler(handler)
        for handler in self._handlers:
            handler.flush()
            handler.close()
        self._handlers.clear()
        for name, level in self._saved_levels.items():
            logging.getLogger(name).setLevel(level)
        self._saved_levels.clear()

    def __exit__(self, *exc: object) -> None:
        self.close()
