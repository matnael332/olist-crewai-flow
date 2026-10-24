"""Page scaffolding shared by every page: header, sections, chart cards, empty state, footer."""

from __future__ import annotations

import altair as alt
import streamlit as st

REPO_URL = "https://github.com/matnael332/pulse"

# Only what the theme cannot express: a readable content width and a tighter top margin.
_CSS = """
<style>
[data-testid="stMainBlockContainer"] { max-width: 1240px; padding-top: 2.4rem; padding-bottom: 3rem; }
[data-testid="stMetricValue"] { font-weight: 650; }
</style>
"""


def apply_global_style() -> None:
    st.html(_CSS)


def page_header(title: str, subtitle: str | None = None, kicker: str | None = None) -> None:
    if kicker:
        st.caption(f":blue[**{kicker.upper()}**]")
    st.title(title, anchor=False)
    if subtitle:
        st.markdown(f":gray[{subtitle}]")


def section(title: str, caption: str | None = None) -> None:
    st.space("medium")
    st.subheader(title, anchor=False)
    if caption:
        st.caption(caption)


def chart_card(title: str, chart: alt.TopLevelMixin, takeaway: str | None = None) -> None:
    """A bordered card: bold title, interactive chart, one-line takeaway."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.altair_chart(chart, width="stretch")
        if takeaway:
            st.caption(f":material/lightbulb: {takeaway}")


def empty_state(message: str) -> None:
    st.info(message, icon=":material/inbox:")


def footer() -> None:
    st.space("medium")
    st.divider()
    st.caption(
        "Data: Olist Brazilian E-Commerce Public Dataset (CC BY-NC-SA 4.0, non-commercial "
        f"coursework). Every number on this page is read from the committed pipeline artifacts. "
        f"[Source code]({REPO_URL})"
    )
