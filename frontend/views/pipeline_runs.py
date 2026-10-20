"""Pipeline runs: stages, quality gates, LLM usage and logs of every run of the Flow."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from components import cards, charts, layout
from components.format import num
from services import artifacts as art

layout.page_header(
    "Pipeline runs",
    "Every run of the CrewAI Flow: which stages ran, what the three gates decided, how many LLM "
    "calls it made, and the full log. Failed runs are never promoted, so they appear only here.",
    kicker="Engineering",
)
runs = art.list_runs()
if not runs:
    layout.empty_state("No runs found.")
    st.stop()

labels = [f"{r.get('run_id')}  ·  {r.get('status')}  ·  {r['_source']}" for r in runs]
choice = st.selectbox("Run", range(len(runs)), format_func=lambda i: labels[i])
run = runs[choice]
folder = Path(run["_dir"])
report_dir = folder if (folder / "validation_report.json").exists() else folder / "artifacts"
manifest = art.load_json("run_manifest.json", folder) or {}


def _duration(started: str | None, finished: str | None) -> str:
    try:
        seconds = (
            datetime.fromisoformat(finished) - datetime.fromisoformat(started)
        ).total_seconds()
    except (TypeError, ValueError):
        return "n/a"
    return f"{int(seconds // 60)} min {int(seconds % 60)} s"


@st.fragment(run_every="3s" if run.get("status") == "running" else None)
def show_status() -> None:
    status = art.load_json("run_status.json", folder) or run
    state = status.get("status", "unknown")
    with st.container(horizontal=True, gap="small"):
        cards.badge(
            state, ok=True if state == "success" else (None if state == "running" else False)
        )
        cards.badge(f"mode: {status.get('mode', 'n/a')}", icon=":material/tune:")
        model = manifest.get("run", {}).get("llm_model")
        if model and status.get("mode") == "llm":
            cards.badge(model, icon=":material/smart_toy:")
        cards.badge(
            _duration(status.get("started_at"), status.get("finished_at")), icon=":material/timer:"
        )
        if status.get("inject_fault"):
            cards.badge(f"fault injected: {status['inject_fault']}", ok=False)
    if state not in {"success", "running"}:
        st.error(f"Failed at **{status.get('stage')}**")
        for error in status.get("errors", []):
            st.markdown(f"- `{error}`")
    for warning in status.get("warnings", []):
        st.warning(warning)


show_status()

# Stages -----------------------------------------------------------------------------------------
log = art.run_log(folder)
stages = art.stage_durations(log)
layout.section("Stages", "Seconds per Flow stage, from the STAGE lines of pipeline.log")
if stages:
    left, right = st.columns([3, 2])
    with left:
        layout.chart_card("Stage durations", charts.stage_timeline(stages))
    with right, st.container(border=True, height="stretch"):
        st.markdown("**Steps completed**")
        done = (art.load_json("run_status.json", folder) or run).get("stages_done", [])
        st.markdown("\n".join(f"{i}. `{name}`" for i, name in enumerate(done, start=1)) or "none")
else:
    st.caption("No stage timings in this run's log.")

# Gates ----------------------------------------------------------------------------------------
layout.section("Quality gates", "A run is promoted only when all three pass")
gates = art.gate_summary(art.validation(report_dir))
if gates:
    columns = st.columns(len(gates))
    for column, (name, gate) in zip(columns, gates.items(), strict=True):
        with column.container(border=True):
            st.markdown(f"**{name}**")
            cards.badge(
                f"{'PASS' if gate['passed'] else 'FAIL'} · {gate['n_passed']}/{gate['n_total']}",
                ok=gate["passed"],
            )
            with st.expander("Checks"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Check": c["name"], "OK": c["passed"], "Detail": c["detail"]}
                            for c in gate["checks"]
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                    column_config={"OK": st.column_config.CheckboxColumn()},
                )
else:
    st.caption("No validation report for this run.")

# LLM usage ------------------------------------------------------------------------------------
usage = manifest.get("token_usage") or {}
if usage:
    layout.section("LLM usage", "Gemini calls made by the two crews in this run")
    c1, c2, c3, c4 = st.columns(4)
    cards.kpi(c1, "LLM calls", num(usage.get("successful_requests")))
    cards.kpi(c2, "Prompt tokens", num(usage.get("prompt_tokens")))
    cards.kpi(c3, "Completion tokens", num(usage.get("completion_tokens")))
    cards.kpi(c4, "Total tokens", num(usage.get("total_tokens")))
    sources = manifest.get("narrative_sources") or {}
    if sources:
        st.caption(
            "Narrative sources: "
            + ", ".join(f"`{name}` {source}" for name, source in sorted(sources.items()))
        )

# Log ------------------------------------------------------------------------------------------
layout.section("Log", "pipeline.log; secrets are redacted before anything is written")
if log:
    lines = log.splitlines()
    c1, c2, c3 = st.columns([2, 2, 1])
    levels = c1.pills(
        "Level",
        ["INFO", "WARNING", "ERROR"],
        selection_mode="multi",
        default=["INFO", "WARNING", "ERROR"],
    ) or ["INFO", "WARNING", "ERROR"]
    needle = c2.text_input("Filter", placeholder="e.g. GATE, HANDOFF, CREW, LLM call")
    tail = c3.number_input("Last lines", 50, 5000, 300, step=50)
    shown = [
        line
        for line in lines
        if (art.log_level(line) or "INFO") in levels
        and (not needle or needle.lower() in line.lower())
    ]
    st.code("\n".join(shown[-int(tail) :]) or "(no matching lines)", language=None, height=420)
else:
    st.caption("No pipeline.log for this run.")

if manifest:
    with st.expander("Run manifest: environment, versions and file hashes"):
        st.json(manifest, expanded=1)
layout.footer()
