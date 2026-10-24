"""Check that the numbers in an answer come from the tool results (ADR-11 applied to chat).

Percentages ("11.9%") and decimals ("0.733") in the answer must appear verbatim in what the tools
returned; plain integers (counts, dates) are not checked. A non-empty result is shown to the
operator as "not found in the sources", never silently dropped.
"""

from __future__ import annotations

import json
import re

PERCENT = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d+)?)\s?%")
DECIMAL = re.compile(r"(?<![\w.])(0\.\d{2,4})(?![\d])")


def _percentages(text: str) -> set[str]:
    return {f"{m.group(1)}%" for m in PERCENT.finditer(text)}


def ungrounded(answer: str, tool_results: list) -> list[str]:
    """Numbers in ``answer`` that none of ``tool_results`` contains, in order of appearance."""
    blob = json.dumps(tool_results, ensure_ascii=False, default=str)
    known_percentages = _percentages(blob)
    missing: list[str] = []
    for value in [f"{m.group(1)}%" for m in PERCENT.finditer(answer)]:
        if value not in known_percentages and value not in missing:
            missing.append(value)
    for m in DECIMAL.finditer(answer):
        value = m.group(1)
        if value not in blob and value not in missing:
            missing.append(value)
    return missing
