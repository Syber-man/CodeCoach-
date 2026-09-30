import uuid

import streamlit as st

from agent import ask
from knowledge_base import DOCS

st.set_page_config(page_title="CodeCoach", page_icon="🧠")
st.title("🧠 CodeCoach")
st.caption("Agentic AI assistant for competitive programming. Answers only from verified notes and live tools.")

with st.sidebar:
    st.subheader("Topics covered")
    for d in DOCS:
        st.write(f"• {d['topic']}")
    st.subheader("Live tools")
    st.write("• Codeforces rating lookup\n• Complexity checker\n• Date and time")
    if st.button("New chat"):
        st.session_state.clear()
        st.rerun()

if "thread" not in st.session_state:
    st.session_state.thread = str(uuid.uuid4())
    st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            with st.expander("Details"):
                st.write(m["meta"])

if q := st.chat_input("Ask about an algorithm, a complexity, or a Codeforces handle..."):
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            r = ask(q, st.session_state.thread)
        st.markdown(r["answer"])
        meta = {"route": r["route"], "faithfulness": r["score"],
                "sources": r["sources"], "seconds": r["seconds"]}
        with st.expander("Details"):
            st.write(meta)
    st.session_state.chat.append({"role": "assistant", "content": r["answer"], "meta": meta})
