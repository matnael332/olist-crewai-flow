"""One number style for every page."""

from __future__ import annotations


def pct(value: float | None, digits: int = 1) -> str:
    """0.0679 -> '6.8%'."""
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 0) -> str:
    """96203 -> '96,203'."""
    return "n/a" if value is None else f"{value:,.{digits}f}"


def brl(value: float | None, compact: bool = False) -> str:
    """13179777.95 -> 'BRL 13.18M' (compact) or 'BRL 13,179,777.95'."""
    if value is None:
        return "n/a"
    if compact and abs(value) >= 1_000_000:
        return f"BRL {value / 1_000_000:.2f}M"
    if compact and abs(value) >= 1_000:
        return f"BRL {value / 1_000:.1f}k"
    return f"BRL {value:,.2f}"


def points(delta: float) -> str:
    """Difference of two probabilities in percentage points: -0.125 -> '-12.5 pts'."""
    return f"{delta * 100:+.1f} pts"


def times(value: float | None) -> str:
    """3.1719 -> '3.2x'."""
    return "n/a" if value is None else f"{value:.1f}x"


def month_name(month: str) -> str:
    """'2017-11' -> 'Nov 2017'."""
    names = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    year, mm = month.split("-")[:2]
    return f"{names[int(mm) - 1]} {year}"
