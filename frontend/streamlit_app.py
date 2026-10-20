"""Olist late-delivery AI workflow: dashboard entrypoint (Streamlit Community Cloud main file).

uv run streamlit run frontend/streamlit_app.py
"""

from pathlib import Path

import streamlit as st

import _bootstrap  # noqa: F401  (backend/ on sys.path for the scoring code)
from components import layout

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(
    page_title="Olist delivery risk",
    page_icon=str(ASSETS / "icon.svg"),
    layout="wide",
    initial_sidebar_state="auto",  # open on desktop, collapsed on phones
)
layout.apply_global_style()
st.logo(
    str(ASSETS / "logo.svg"),
    size="large",
    link=layout.REPO_URL,
    icon_image=str(ASSETS / "icon.svg"),
)

pages = {
    "Operate": [
        st.Page("views/copilot.py", title="Ops copilot", icon=":material/forum:"),
    ],
    "Business": [
        st.Page(
            "views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True
        ),
        st.Page("views/insights.py", title="Insights", icon=":material/insights:"),
        st.Page("views/scoring.py", title="Risk scoring", icon=":material/speed:"),
    ],
    "Data science": [
        st.Page("views/model.py", title="Model", icon=":material/model_training:"),
        st.Page("views/data_contract.py", title="Data contract", icon=":material/fact_check:"),
    ],
    "Engineering": [
        st.Page("views/pipeline_runs.py", title="Pipeline runs", icon=":material/account_tree:"),
    ],
}
navigation = st.navigation(pages)

with st.sidebar:
    st.caption(
        "Predicts at checkout which Olist orders will arrive late. A CrewAI Flow hands off from a "
        "Data Analyst crew to a Data Scientist crew, with three quality gates in between."
    )
    st.caption(f"[Source code on GitHub]({layout.REPO_URL})")

navigation.run()
