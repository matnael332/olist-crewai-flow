"""KPI cards, status badges and insight cards."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st


def kpi(
    target,
    label: str,
    value: str,
    help: str | None = None,
    delta: str | float | None = None,
    delta_color: str = "normal",
    chart: Sequence[float] | None = None,
    chart_type: str = "line",
    height: int | str = "content",
) -> None:
    """A bordered metric; ``target`` is a column or container."""
    target.metric(
        label,
        value,
        delta=delta,
        delta_color=delta_color,
        help=help,
        border=True,
        chart_data=list(chart) if chart is not None else None,
        chart_type=chart_type,
        height=height,
    )


def badge(text: str, ok: bool | None = None, icon: str | None = None, target=None) -> None:
    """Green for passed, red for failed, gray for neutral facts; the text always says which."""
    color = "gray" if ok is None else ("green" if ok else "red")
    if icon is None and ok is not None:
        icon = ":material/check_circle:" if ok else ":material/cancel:"
    (target or st).badge(text, icon=icon, color=color)


def insight(target, title: str, value: str, body: str, icon: str) -> None:
    """A card with an icon, a headline number and one sentence of context."""
    with target.container(border=True, height="stretch"):
        st.markdown(f"{icon} **{title}**")
        st.markdown(
            f"<div style='font-size:1.45rem;font-weight:650;line-height:1.3'>{value}</div>",
            unsafe_allow_html=True,
        )
        st.caption(body)
