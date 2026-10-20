from pathlib import Path

import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401
from services.artifacts import list_runs, load_json, load_text

st.title("Runs and logs")
st.caption(
    "The committed (promoted) run, plus any local runs in runs/ (failed runs are never promoted, "
    "so they only appear here)."
)
runs = list_runs()
if not runs:
    st.info("No runs found.")
    st.stop()

labels = [f"{r.get('run_id')}  |  {r.get('status')}  |  {r['_source']}" for r in runs]
choice = st.selectbox("Run", range(len(runs)), format_func=lambda i: labels[i])
run = runs[choice]
folder = Path(run["_dir"])


@st.fragment(run_every="3s" if run.get("status") == "running" else None)
def show_status() -> None:
    status = load_json("run_status.json", folder) or run
    state = status.get("status")
    if state == "success":
        st.success(f"success: {', '.join(status.get('stages_done', []))}")
    elif state == "running":
        st.info(f"running: stage {status.get('stage')} (since {status.get('stage_started_at')})")
    else:
        st.error(f"{state} at {status.get('stage')}")
        for error in status.get("errors", []):
            st.markdown(f"- `{error}`")
    for warning in status.get("warnings", []):
        st.warning(warning)


show_status()
report_dir = folder if (folder / "validation_report.json").exists() else folder / "artifacts"
report = load_json("validation_report.json", report_dir)
if report:
    checks = [
        {
            "gate": gate,
            "check": c["name"],
            "result": "PASS" if c["passed"] else "FAIL",
            "detail": c["detail"],
        }
        for gate, r in report["gates"].items()
        for c in r["checks"]
    ]
    st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch", height=320)
log = load_text("logs/pipeline.log", folder) or load_text("pipeline.log", folder)
if log:
    st.subheader("pipeline.log")
    st.code("\n".join(log.splitlines()[-200:]), language=None)
manifest = load_json("run_manifest.json", folder)
if manifest:
    with st.expander("run_manifest.json"):
        st.json(manifest, expanded=False)
