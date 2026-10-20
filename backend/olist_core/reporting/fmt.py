"""Number formatting shared by templates and renderers (one place, consistent output)."""

from __future__ import annotations


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def num(value: float | int | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def brl(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1_000_000:
        return f"BRL {value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"BRL {value / 1_000:.1f}k"
    return f"BRL {value:.2f}"


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- (none)"
