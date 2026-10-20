import streamlit as st
import streamlit.components.v1 as components

import _bootstrap  # noqa: F401
from services.artifacts import load_text

st.title("EDA report and business insights")
tab_report, tab_insights = st.tabs(["eda_report.html", "insights.md"])
with tab_report:
    html = load_text("eda_report.html")
    if html:
        components.html(html, height=1400, scrolling=True)
        st.download_button("Download eda_report.html", html, file_name="eda_report.html")
    else:
        st.info("No EDA report in artifacts/ yet.")
with tab_insights:
    insights = load_text("insights.md")
    st.markdown(insights or "No insights yet.")
