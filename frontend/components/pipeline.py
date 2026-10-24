"""The four-step 'how it works' stepper, built from the run status and the validation report."""

from __future__ import annotations

import streamlit as st

from components.cards import badge

STEPS = [
    {
        "title": "Analyze",
        "body": "Data Analyst crew (6 agents): clean the raw tables, explore, write insights "
        "and the dataset contract.",
        "gate": "gate_contract",
        "gate_label": "Contract gate",
    },
    {
        "title": "Engineer features",
        "body": "Data Scientist crew, phase 1: decision-time features, validated against the "
        "contract.",
        "gate": "gate_required_features",
        "gate_label": "Feature gate",
    },
    {
        "title": "Train and document",
        "body": "Data Scientist crew, phase 2: three models compared on a later window, "
        "evaluation report and model card.",
        "gate": "gate_artifacts",
        "gate_label": "Artifacts gate",
    },
    {
        "title": "Publish",
        "body": "Only a run that passes every gate replaces artifacts/, which this dashboard and "
        "the API read.",
        "gate": None,
        "gate_label": "Promoted",
    },
]


def stepper(gates: dict[str, dict], promoted: bool) -> None:
    """``gates``: gate name -> {'passed', 'n_passed', 'n_total'} (see services.artifacts)."""
    columns = st.columns(len(STEPS))
    for number, (column, step) in enumerate(zip(columns, STEPS, strict=True), start=1):
        with column.container(border=True, height="stretch"):
            st.caption(f"STEP {number}")
            st.markdown(f"**{step['title']}**")
            if step["gate"]:
                gate = gates.get(step["gate"])
                if gate:
                    badge(
                        f"{step['gate_label']} {gate['n_passed']}/{gate['n_total']}",
                        ok=gate["passed"],
                    )
                else:
                    badge(f"{step['gate_label']}: no report", ok=None)
            else:
                badge("Published" if promoted else "Not promoted", ok=promoted)
            st.caption(step["body"])
