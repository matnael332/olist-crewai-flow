"""What one copilot turn produces, and the per-chat state the tools share."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Conversation:
    """Per-chat state: the attached CSV (if any) and rich outputs the UI renders (tables, files).

    ``attachment``: {"name": str, "records": list[dict], "ignored": list[str]} or
    {"name": str, "error": str} when the file could not be read.
    ``outputs``: {"kind": "batch", "title": str, "result": BatchResult, "backtest": dict | None}.
    """

    attachment: dict | None = None
    outputs: list[dict] = field(default_factory=list)


@dataclass
class ToolCall:
    name: str
    args: dict
    result: dict


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    offline: bool = False
    error: str | None = None


class Recorder:
    """Wraps tool functions so every call (name, arguments, result) is kept for the UI."""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def wrap(self, function: Callable) -> Callable:
        @functools.wraps(function)
        def recorded(*args, **kwargs):
            result = function(*args, **kwargs)
            self.calls.append(ToolCall(function.__name__, dict(kwargs), result))
            return result

        return recorded

    def sources(self) -> list[str]:
        """Source files of every call, in first-seen order (the deterministic Sources line)."""
        seen: list[str] = []
        for call in self.calls:
            for source in call.result.get("source", []) if isinstance(call.result, dict) else []:
                if source not in seen:
                    seen.append(source)
        return seen
