"""Ops copilot: a WhatsApp-style chat with the system the agents built (read-only)."""

import streamlit as st

from components import cards, layout
from services import copilot as chat

layout.page_header(
    "Ops copilot",
    "Ask about the agents' runs, gates, model and data, or score records. It reads what the "
    "agents produced and never runs or changes anything.",
    kicker="Operate",
)
if not chat.unlocked():
    chat.passcode_gate()
    st.stop()

bot = chat.bot()
with st.container(horizontal=True, gap="small", vertical_alignment="center"):
    if bot.available:
        cards.badge(f"Gemini online · {bot.model}", ok=True, icon=":material/smart_toy:")
    else:
        cards.badge("Offline: instant answers only", icon=":material/cloud_off:")
    cards.badge(
        f"{chat.remaining()} of {chat.message_limit()} messages left", icon=":material/forum:"
    )
    if st.button("New chat", icon=":material/refresh:", type="tertiary"):
        chat.reset()
        st.rerun()

with st.expander("What can I ask?", icon=":material/help:"):
    st.markdown(
        "- **Runs and gates:** *How did the last run go?* · *Why did gate_contract pass?* · "
        "*Show the HANDOFF line*\n"
        "- **Model and data:** *Which model was selected and why?* · *What does the model card "
        "say about limitations?* · *Which features are forbidden?*\n"
        "- **Files:** attach a CSV of records (up to 5,000 rows, one column per model feature) "
        "and get a ranked list and an action list."
    )
    st.download_button(
        "CSV template",
        chat.template_csv(),
        "records_template.csv",
        "text/csv",
        icon=":material/download:",
    )

history = chat.messages()
if not history:
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        st.markdown(
            "Hi! I keep an eye on the pipeline's agents. Pick a quick reply, ask a question, "
            "or attach a CSV of records to score."
        )
for index, message in enumerate(history):
    chat.render(message, index)

quick = st.pills(
    "Quick replies",
    list(chat.QUICK),
    key=f"quick-{len(history)}",
    label_visibility="collapsed",
)
prompt = st.chat_input(
    "Ask about runs, gates, the model or the data. Attach a CSV to score it.",
    accept_file=True,
    file_type=["csv"],
    max_chars=2000,
)

if quick or prompt:
    if chat.remaining() <= 0:
        st.warning(
            f"This session used all {chat.message_limit()} messages. Reload the page to start a "
            "new session."
        )
        st.stop()
    text = (prompt.text if prompt else "") or ""
    files = list(prompt.files) if prompt and prompt.files else []
    uploaded = files[0] if files else None
    shown = quick or text or "Score my attached file"
    with st.chat_message("user"):
        st.markdown(shown)
    with (
        st.chat_message("assistant", avatar=":material/smart_toy:"),
        st.spinner("Thinking..."),
    ):
        if quick:
            chat.ask(quick, quick=chat.QUICK[quick])
        elif uploaded is not None and not text.strip():
            chat.ask(shown, quick="attachment", uploaded=uploaded)
        else:
            chat.ask(text, uploaded=uploaded)
    st.rerun()
