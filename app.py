"""
app.py - the chatbot as a web app.

    streamlit run app.py
"""
import streamlit as st

import config
from rag import HandbookRAG

st.set_page_config(page_title="IITM Handbook Bot", page_icon="📘")


@st.cache_resource          # load the model + DB only once, not on every click
def load_rag():
    return HandbookRAG()


rag = load_rag()

# ---------- Sidebar: knobs to experiment with ----------
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, config.TOP_K)
    min_score = st.slider("Minimum score", 0.0, 1.0, config.MIN_SCORE, 0.05)
    show_sources = st.toggle("Show sources", value=True)
    if rag.llm is None:
        st.warning("No GROQ_API_KEY found - showing passages only. "
                   "Add your key to the .env file.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("📘 IIT Madras Handbook Bot")
st.caption("Answers come only from the student handbook, with page citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Show past messages ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- New question ----------
if question := st.chat_input("e.g. What happens if I fail a course?"):
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages]
    with st.spinner("Searching the handbook ..."):
        result = rag.answer(question, history, k=top_k, min_score=min_score)

    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        if show_sources and result["sources"]:
            with st.expander("📄 Sources (what the bot read)"):
                for h in result["sources"]:
                    st.markdown(f"**Page {h['page']}** · score `{h['score']:.3f}`")
                    st.caption(h["text"])

    st.session_state.messages += [{"role": "user", "content": question},
                                  {"role": "assistant", "content": result["answer"]}]
