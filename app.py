import streamlit as st
from pathlib import Path
import tempfile
import os

from ingestion.parser import extract
from vectorstore.chunker import chunk_document
from vectorstore.embedder import embed_chunks
from vectorstore.milvus_client import init_collection, upsert_chunks, delete_document
from graph.graph import run

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Doc Chat",
    page_icon="📄",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────

def init_session():
    defaults = {
        "doc_id":       None,
        "filename":     None,
        "full_text":    None,
        "chat_history": [],      # list of {role, content}
        "edit_history": [],      # list of edit records
        "current_text": "",
        "indexed":      False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()
init_collection()

# ── Sidebar — file upload ─────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 Doc Chat")
    st.caption("Upload a document and chat with it.")

    uploaded = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "pptx", "xlsx"],
        help="Supported: PDF, DOCX, PPTX, XLSX",
    )

    if uploaded:
        # only re-index if a new file is uploaded
        if uploaded.name != st.session_state.get("filename"):

            with st.spinner("Extracting and indexing..."):
                # save to temp file so parsers can read from path
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                # clear previous doc from Milvus if one existed
                if st.session_state.doc_id:
                    delete_document(st.session_state.doc_id)

                # extract → chunk → embed → index
                doc    = extract(tmp_path)
                chunks = chunk_document(doc)
                chunks = embed_chunks(chunks)
                upsert_chunks(chunks)
                os.unlink(tmp_path)

                # reset session for new doc
                st.session_state.doc_id       = doc["doc_id"]
                st.session_state.filename     = uploaded.name
                st.session_state.full_text    = doc["full_text"]
                st.session_state.chat_history = []
                st.session_state.edit_history = []
                st.session_state.current_text = doc["full_text"]
                st.session_state.indexed      = True

            st.success(f"Ready — {uploaded.name}")

    # doc info
    if st.session_state.indexed:
        st.divider()
        st.markdown("**Document**")
        st.caption(st.session_state.filename)
        st.markdown("**Edit history**")
        if st.session_state.edit_history:
            for i, e in enumerate(st.session_state.edit_history, 1):
                st.caption(f"{i}. {e['section']} — {e['instruction'][:40]}...")
        else:
            st.caption("No edits yet.")

        # clear button
        st.divider()
        if st.button("Clear & upload new doc", use_container_width=True):
            if st.session_state.doc_id:
                delete_document(st.session_state.doc_id)
            for key in ["doc_id", "filename", "full_text", "chat_history",
                        "edit_history", "current_text", "indexed"]:
                st.session_state[key] = None if key in ["doc_id", "filename", "full_text"] else \
                                        [] if key in ["chat_history", "edit_history"] else \
                                        "" if key == "current_text" else False
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────

if not st.session_state.indexed:
    st.markdown("## Welcome to Doc Chat")
    st.markdown(
        "Upload a **PDF, DOCX, PPTX, or XLSX** file from the sidebar to get started.\n\n"
        "You can:\n"
        "- Ask questions about the document\n"
        "- Request a summary\n"
        "- Ask for explanations in simple terms\n"
        "- Edit sections with natural language instructions\n"
    )
    st.stop()

st.markdown(f"### {st.session_state.filename}")
st.caption(f"doc_id: {st.session_state.doc_id}")
st.divider()

# ── Chat history display ──────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

query = st.chat_input("Ask a question, request a summary, or give an edit instruction...")

if query:
    # show user message immediately
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # run the graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run(
                query=query,
                doc_id=st.session_state.doc_id,
                filename=st.session_state.filename,
                current_text=st.session_state.current_text,
                edit_history=st.session_state.edit_history,
            )

        response     = result["response"]
        edit_history = result["edit_history"]
        current_text = result["current_text"]

        st.markdown(response)

    # update session state
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.edit_history = edit_history
    st.session_state.current_text = current_text
    st.rerun()