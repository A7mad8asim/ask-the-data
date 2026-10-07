"""Ask-the-Data chat app.   Run:  streamlit run app.py"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from askdata.config import DEFAULT_MODELS, Settings
from askdata.pipeline import Answer, Assistant

st.set_page_config(page_title="Ask-the-Data", page_icon=":material/query_stats:", layout="wide")

EXAMPLES = [
    "What was the no-show rate at each clinic in 2025?",
    "How many visits were there in each month of 2024?",
    "ما نسبة إكمال المتابعات لكل مجموعة جنسية، للمتابعات المستحقة في 2025؟",
    "كم كانت نسبة الغياب عن المواعيد في رمضان 2025؟",
    "Which clinic had the biggest increase in no-show rate during Ramadan 2025 compared with February 2025?",
    "List the IDs of all patients with an HbA1c above 10.",
]
PROVIDERS = {
    "ollama": "Local model (Ollama)",
    "anthropic": "Claude API",
    "oracle": "Gold SQL (demo, eval questions only)",
}


@st.cache_resource(show_spinner="Opening the database ...")
def get_assistant(provider: str, model: str) -> Assistant:
    settings = Settings.from_env().with_overrides(llm_provider=provider, llm_model=model)
    return Assistant(settings)


def rtl_block(text: str, language: str) -> None:
    direction = "rtl" if language == "ar" else "ltr"
    body = html.escape(text).replace("\n", "<br>")
    st.markdown(f"<div dir='{direction}' style='font-size:1.05rem;line-height:1.6'>{body}</div>", unsafe_allow_html=True)


def render_answer(ans: Answer, key: str) -> None:
    if ans.status == "refused":
        st.warning(ans.message, icon=":material/shield:")
    elif ans.status == "error":
        st.error(ans.message, icon=":material/error:")
    if ans.status != "ok":
        tried = [a for a in ans.attempts if a.sql]
        if tried:
            with st.expander("What the model tried"):
                for a in tried:
                    st.code(a.sql, language="sql")
                    st.caption(f"Rejected ({a.stage}): {a.error}")
        return

    rtl_block(ans.text, ans.language)
    notes = [f"{ans.latency_s:.1f} s", f"{len(ans.attempts)} attempt" + ("s" if len(ans.attempts) > 1 else "")]
    if ans.grounded is False:
        notes.append("the model's wording used numbers not in the table, so a plain summary is shown")
    st.caption(" · ".join(notes))
    if ans.suppressed_rows:
        st.info(
            f"{ans.suppressed_rows} row(s) describe fewer than 10 patients; their values are shown as <10.",
            icon=":material/visibility_off:",
        )

    shown = ans.display_table
    names = (["Chart"] if ans.chart else []) + ["Table", "SQL"]
    tabs = dict(zip(names, st.tabs(names)))
    if ans.chart:
        with tabs["Chart"]:
            data = ans.table.copy()
            for col in ans.chart["y"]:
                data[col] = pd.to_numeric(data[col], errors="coerce")
            x = ans.chart["x"]
            if ans.chart["type"] == "line":
                st.line_chart(data, x=x, y=ans.chart["y"])
            else:
                data[x] = data[x].astype(str)
                st.bar_chart(data, x=x, y=ans.chart["y"], horizontal=len(data) > 6)
    with tabs["Table"]:
        st.dataframe(shown, hide_index=True, width="stretch")
        if ans.truncated:
            st.caption("Only the first rows are shown.")
        st.download_button(
            "Download CSV",
            shown.to_csv(index=False).encode("utf-8-sig"),
            file_name="ask-the-data.csv",
            mime="text/csv",
            key=f"csv-{key}",
            icon=":material/download:",
        )
    with tabs["SQL"]:
        st.code(ans.sql, language="sql")


# ---------------------------------------------------------------------------- sidebar

defaults = Settings.from_env()
with st.sidebar:
    st.subheader("Ask-the-Data")
    st.caption("اسأل البيانات · bilingual analytics for a fictional Doha clinic network")
    provider = st.selectbox(
        "Model",
        list(PROVIDERS),
        index=list(PROVIDERS).index(defaults.llm_provider) if defaults.llm_provider in PROVIDERS else 0,
        format_func=PROVIDERS.get,
    )
    model = st.text_input(
        "Model name",
        value=defaults.llm_model if provider == defaults.llm_provider else DEFAULT_MODELS[provider],
        disabled=provider == "oracle",
    )
    st.divider()
    st.caption("Try a question")
    for i, q in enumerate(EXAMPLES):
        if st.button(q, key=f"ex-{i}", width="stretch"):
            st.session_state.pending = q
    st.divider()
    st.caption(
        "Synthetic data: 60,000 patients, 8 clinics, 2023-2025. Answers are aggregated; "
        "groups under 10 patients are hidden. Operational analytics only, not medical advice."
    )
    if st.button("Clear chat", icon=":material/delete_sweep:"):
        st.session_state.history = []

# ---------------------------------------------------------------------------- chat

st.title("Ask-the-Data")
st.caption("Ask about appointments, visits, screenings, lab results and follow-ups, in Arabic or English.")

try:
    assistant = get_assistant(provider, model)
except FileNotFoundError as e:
    st.error(f"{e}", icon=":material/database:")
    st.stop()

history: list[Answer] = st.session_state.setdefault("history", [])
for i, ans in enumerate(history):
    with st.chat_message("user"):
        rtl_block(ans.question, ans.language)
    with st.chat_message("assistant"):
        render_answer(ans, key=str(i))

question = st.chat_input("Ask a question in Arabic or English ...") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        from askdata.answers import detect_language

        rtl_block(question, detect_language(question))
    with st.chat_message("assistant"):
        with st.spinner("Writing the SQL, checking it and running it ..."):
            ans = assistant.ask(question)
        render_answer(ans, key=str(len(history)))
    history.append(ans)
