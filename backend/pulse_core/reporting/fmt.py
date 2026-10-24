"""Number formatting shared by templates and renderers (one place, consistent output)."""

from __future__ import annotations

import re

# Keys whose numbers are fractions in [0, 1]: positive_rate, share, prevalence, precision, ...
_PCT_KEY = re.compile(r"(^|_)(rate|share|prevalence|coverage)(_|$)|^(precision|recall)(_|$)")


def as_percentages(value, pct: bool = False):
    """Format rate-like numbers as percentage strings ("6.8%") before an LLM sees them.

    LLMs copy these strings instead of converting fractions themselves, which removes rounding
    slips such as writing 9.3% for 0.092474. Children of a rate-like key inherit it; only floats
    in [0, 1] are converted. Used by the crews' tools and by the ops copilot's tools.
    """
    if isinstance(value, dict):
        return {
            k: as_percentages(v, pct or bool(_PCT_KEY.search(str(k)))) for k, v in value.items()
        }
    if isinstance(value, list):
        return [as_percentages(v, pct) for v in value]
    if pct and isinstance(value, float) and 0.0 <= value <= 1.0:
        return f"{value * 100:.1f}%"
    return value


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def num(value: float | int | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def key_number(label: str, value) -> str:
    """Format one key number by its label: rates and shares as %, counts with separators."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and _PCT_KEY.search(label.lower().replace(" ", "_")):
        return pct(value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return num(value)
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- (none)"
