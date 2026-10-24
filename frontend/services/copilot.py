"""Session state, access control and rendering for the Ops copilot chat page.

The brain lives in the backend (pulse_copilot). This module only keeps the chat in
st.session_state, checks the passcode, enforces the message cap, reads attachments and draws
messages. Keys and the passcode come from environment variables (`.env` on the server).
"""

from __future__ import annotations

import hmac
import os

import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401  (backend/ on sys.path)
from components.format import pct
from pulse_api.batch import BatchError, frame_to_records, read_records_csv
from pulse_api.risk import ACTIONS
from pulse_copilot.agent import Copilot
from pulse_copilot.answer import Answer, Conversation
from pulse_copilot.prompts import QUICK_REPLIES
from pulse_copilot.quick import quick_answer
from pulse_copilot.workspace import Workspace
from services.artifacts import ARTIFACTS

MAX_ATTEMPTS = 5
MAX_FILE_BYTES = 2 * 1024 * 1024
LEVEL_LABELS = {"high": "🔴 High", "elevated": "🟠 Elevated", "normal": "🟢 Normal"}


def message_limit() -> int:
    try:
        return max(1, int(os.getenv("COPILOT_MAX_MESSAGES", "25")))
    except ValueError:
        return 25


def _state() -> dict:
    if "copilot" not in st.session_state:
        st.session_state["copilot"] = {
            "messages": [],
            "conversation": Conversation(),
            "used": 0,
            "attempts": 0,
            "unlocked": False,
            "bot": Copilot(Workspace.from_env(ARTIFACTS)),
        }
    return st.session_state["copilot"]


def bot() -> Copilot:
    return _state()["bot"]


def messages() -> list[dict]:
    return _state()["messages"]


def remaining() -> int:
    return message_limit() - _state()["used"]


def reset() -> None:
    state = _state()
    state["messages"], state["conversation"] = [], Conversation()


# Access ---------------------------------------------------------------------------------------


def passcode_required() -> bool:
    return bool(os.getenv("COPILOT_PASSCODE"))


def unlocked() -> bool:
    return not passcode_required() or _state()["unlocked"]


def passcode_gate() -> None:
    """A small form; unlocks the chat for this session when the passcode matches."""
    state = _state()
    with st.form("copilot-passcode", border=True, width=420):
        st.markdown("**This chat runs on a Gemini key.** Enter the passcode to open it.")
        code = st.text_input("Passcode", type="password")
        submitted = st.form_submit_button("Open the copilot", type="primary")
    if not submitted:
        return
    if state["attempts"] >= MAX_ATTEMPTS:
        st.error("Too many attempts in this session. Reload the page to try again.")
        return
    state["attempts"] += 1
    expected = os.getenv("COPILOT_PASSCODE", "")
    if hmac.compare_digest(code.encode(), expected.encode()):
        state["unlocked"] = True
        st.rerun()
    st.error("That passcode is not right.")


# Turns ----------------------------------------------------------------------------------------


def attach(uploaded) -> dict:
    """An uploaded CSV -> the conversation's attachment (records, or a readable error)."""
    data = uploaded.getvalue()
    if len(data) > MAX_FILE_BYTES:
        return {"name": uploaded.name, "error": "the file is larger than 2 MB"}
    bundle = bot().workspace.bundle()
    if not bundle.ok:
        return {"name": uploaded.name, "error": f"model unavailable: {bundle.error}"}
    try:
        records, ignored = frame_to_records(bundle, read_records_csv(bundle, data))
    except BatchError as exc:
        missing = exc.details.get("missing")
        detail = f": {', '.join(missing)}" if missing else ""
        return {"name": uploaded.name, "error": f"{exc.message}{detail}"}
    return {"name": uploaded.name, "records": records, "ignored": ignored}


def ask(text: str, quick: str | None = None, uploaded=None) -> Answer:
    """One turn. Quick replies (and a bare attachment) are answered instantly without an LLM;
    free text goes to Gemini with the tools."""
    state = _state()
    conversation: Conversation = state["conversation"]
    user = {"role": "user", "text": text}
    if uploaded is not None:
        conversation.attachment = attach(uploaded)
        user["file"] = uploaded.name
    state["messages"].append(user)
    state["used"] += 1
    copilot = state["bot"]
    if quick:
        answer = quick_answer(quick, copilot.workspace, conversation)
    else:
        history = [m for m in state["messages"][:-1] if m.get("text")]
        answer = copilot.ask(text, history, conversation)
    state["messages"].append({"role": "assistant", "text": answer.text, "answer": answer})
    return answer


# Rendering ------------------------------------------------------------------------------------


def _table(result) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Record": p["id"][:24] if p["id"] else f"row {p['row']}",
                "Probability": pct(p["probability"]),
                "Risk": LEVEL_LABELS[p["risk_level"]],
            }
            for p in result.predictions
        ]
    )


def render_output(output: dict, key: str) -> None:
    result = output["result"]
    summary = result.summary
    with st.container(border=True):
        st.markdown(f"**{output['title']}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("High", summary["high"])
        c2.metric("Elevated", summary["elevated"])
        c3.metric("Normal", summary["normal"])
        st.caption(f"**High:** {ACTIONS['high']} **Elevated:** {ACTIONS['elevated']}")
        st.dataframe(_table(result).head(15), hide_index=True, width="stretch")
        if result.errors:
            with st.expander(f"{summary['rejected']} rows failed validation"):
                st.dataframe(pd.DataFrame(result.errors), hide_index=True, width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button(
            "Ranked list (CSV)",
            result.to_csv(),
            "scored_records.csv",
            "text/csv",
            key=f"{key}-all",
            icon=":material/download:",
            width="stretch",
        )
        d2.download_button(
            "Action list (CSV)",
            result.action_list().to_csv(index=False, lineterminator="\n"),
            "action_list.csv",
            "text/csv",
            key=f"{key}-actions",
            icon=":material/task_alt:",
            width="stretch",
        )


def render(message: dict, index: int) -> None:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["text"] or "_(file only)_")
            if message.get("file"):
                st.caption(f":material/attach_file: {message['file']}")
        return
    answer: Answer = message["answer"]
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        st.markdown(answer.text)
        notes = []
        if answer.sources:
            notes.append(":material/description: Sources: " + " · ".join(answer.sources))
        if answer.offline and not answer.error:
            notes.append(":material/bolt: instant answer, no LLM")
        if notes:
            st.caption("  \n".join(notes))
        if answer.ungrounded:
            st.caption(
                ":material/warning: Not found in the sources: " + ", ".join(answer.ungrounded)
            )
        for j, output in enumerate(answer.outputs):
            render_output(output, key=f"out-{index}-{j}")


def template_csv() -> str:
    """A header-only CSV with the record fields the model expects (from feature_spec.json)."""
    bundle = bot().workspace.bundle()
    if not bundle.ok:
        return "id\n"
    fields = ([bundle.id_field] if bundle.id_field else []) + list(bundle.validator.features)
    return pd.DataFrame(columns=fields).to_csv(index=False, lineterminator="\n")


QUICK = QUICK_REPLIES
