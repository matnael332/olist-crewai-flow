"""Olist late-delivery AI workflow: dashboard entrypoint.

uv run streamlit run frontend/streamlit_app.py
"""

import streamlit as st

import _bootstrap  # noqa: F401

st.set_page_config(
    page_title="Olist late-delivery AI workflow",
    page_icon=":package:",
    layout="wide",
)

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/data_contract.py", title="Data and contract", icon=":material/table:"),
    st.Page("views/eda_report.py", title="EDA and insights", icon=":material/insights:"),
    st.Page("views/model.py", title="Model", icon=":material/model_training:"),
    st.Page("views/predict.py", title="Predict", icon=":material/online_prediction:"),
    st.Page("views/run_logs.py", title="Runs and logs", icon=":material/terminal:"),
]

with st.sidebar:
    st.markdown("### Olist late-delivery AI workflow")
    st.caption(
        "A CrewAI Flow hands off from a Data Analyst crew to a Data Scientist crew. "
        "Every number shown here comes from the committed pipeline artifacts."
    )

st.navigation(pages).run()
