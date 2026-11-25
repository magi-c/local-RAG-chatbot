from pathlib import Path

import streamlit as st

from rag.loader import load_pdf
from rag.chunker import chunk_pages
from rag.store import add_chunks, search
from rag.answer import client, build_messages, cited_sources
from rag.config import CHAT_MODEL

st.set_page_config(page_title="Chat with your PDF")
st.title("Chat with your PDF")
st.caption(f"Running locally on Ollama · {CHAT_MODEL}")

with st.sidebar:
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded and st.button("Index it"):
        path = Path("data") / uploaded.name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(uploaded.getvalue())
        with st.spinner("Indexing..."):
            chunks = chunk_pages(load_pdf(path))
            add_chunks(chunks)
        st.success(f"Added {len(chunks)} chunks")
    k = st.slider("Chunks to retrieve (k)", 1, 10, 5)
    min_score = st.slider("Minimum similarity", 0.0, 1.0, 0.5)

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    st.chat_message(role).markdown(content)


def stream_tokens(messages):
    for part in client.chat(model=CHAT_MODEL, messages=messages, stream=True,
                            options={"temperature": 0, "num_ctx": 8192}):
        yield part.message.content


if question := st.chat_input("Ask something about the document"):
    st.chat_message("user").markdown(question)
    hits = [h for h in search(question, k=k) if h["score"] >= min_score]
    with st.chat_message("assistant"):
        if hits:
            answer = st.write_stream(stream_tokens(build_messages(question, hits)))
        else:
            answer = "I couldn't find that in the document."
            st.markdown(answer)
        for s in cited_sources(answer, hits):
            with st.expander(f'[{s["n"]}] {s["source"]}, page {s["page"]}'):
                st.write(s["text"])
    st.session_state.history += [("user", question), ("assistant", answer)]