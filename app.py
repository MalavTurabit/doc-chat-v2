import streamlit as st
import requests
from pathlib import Path

API_BASE = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Doc Chat",
    page_icon="📄",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────

def init_session():
    if "session_id" not in st.session_state:
        res = requests.post(f"{API_BASE}/session")
        st.session_state.session_id = res.json()["session_id"]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "docs" not in st.session_state:
        st.session_state.docs = []    # [{doc_id, filename}]

init_session()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 Doc Chat")
    st.caption("Upload documents and chat with all of them.")
    st.page_link("pages/audio_explainer.py", label="🎙️ Audio explainer", icon=None)
    st.divider()
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "pptx", "xlsx", "csv", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        existing_names = {d["filename"] for d in st.session_state.docs}
        new_files = [f for f in uploaded_files if f.name not in existing_names]

        for uploaded in new_files:
            with st.spinner(f"Indexing {uploaded.name}..."):
                res = requests.post(
                    f"{API_BASE}/upload",
                    params={"session_id": st.session_state.session_id},
                    files={"file": (uploaded.name, uploaded.getvalue())},
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.docs.append({
                        "doc_id":   data["doc_id"],
                        "filename": data["filename"],
                        "chunks":   data["chunks"],
                    })
                    st.success(f"Ready — {uploaded.name} ({data['chunks']} chunks)")
                else:
                    st.error(f"Failed to index {uploaded.name}: {res.text}")

    # ── Uploaded docs list ────────────────────────────────────────────────────

    if st.session_state.docs:
        st.divider()
        st.markdown("**Uploaded documents**")
        for d in st.session_state.docs:
            st.caption(f"📄 {d['filename']}  ({d['chunks']} chunks)")

    # ── Download ──────────────────────────────────────────────────────────────

    if st.session_state.docs:
        st.divider()
        st.markdown("**Download**")

        if len(st.session_state.docs) == 1:
            doc_id = st.session_state.docs[0]["doc_id"]
            fname  = st.session_state.docs[0]["filename"]
            res    = requests.get(
                f"{API_BASE}/download/{st.session_state.session_id}",
                params={"doc_id": doc_id},
            )
            st.download_button(
                label=f"Download {Path(fname).stem}_updated.txt",
                data=res.content,
                file_name=f"{Path(fname).stem}_updated.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            # individual downloads
            for d in st.session_state.docs:
                res = requests.get(
                    f"{API_BASE}/download/{st.session_state.session_id}",
                    params={"doc_id": d["doc_id"]},
                )
                st.download_button(
                    label=f"Download {Path(d['filename']).stem}_updated.txt",
                    data=res.content,
                    file_name=f"{Path(d['filename']).stem}_updated.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=d["doc_id"],
                )

            # combined download
            res = requests.get(
                f"{API_BASE}/download/{st.session_state.session_id}"
            )
            st.download_button(
                label="Download all as single TXT",
                data=res.content,
                file_name="all_documents_updated.txt",
                mime="text/plain",
                use_container_width=True,
                key="all",
            )

    # ── Clear session ─────────────────────────────────────────────────────────

    st.divider()
    if st.button("Clear & start over", use_container_width=True):
        requests.delete(f"{API_BASE}/session/{st.session_state.session_id}")
        for key in ["session_id", "chat_history", "docs"]:
            del st.session_state[key]
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────

if not st.session_state.docs:
    st.markdown("## Welcome to Doc Chat")
    st.markdown(
        "Upload one or more **PDF, DOCX, PPTX, XLSX, CSV, or TXT** files.\n\n"
        "- Ask questions across all uploaded documents\n"
        "- Bot will tell you which document the answer came from\n"
        "- Request summaries, explanations, or edits\n"
        "- Download the updated document as plain text\n"
    )
    st.stop()

st.markdown("### Chat with your documents")
doc_names = ", ".join(d["filename"] for d in st.session_state.docs)
st.caption(f"Active documents: {doc_names}")
st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

# ── Chat input ────────────────────────────────────────────────────────────────

query = st.chat_input("Ask anything across your documents...")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = requests.post(
                f"{API_BASE}/chat",
                json={
                    "session_id": st.session_state.session_id,
                    "query":      query,
                },
            )

        if res.status_code == 200:
            data     = res.json()
            response = data["response"]
            sources  = data.get("sources", [])
            st.markdown(response)
            if sources:
                st.caption(f"Sources: {', '.join(sources)}")
        else:
            response = f"Error: {res.text}"
            sources  = []
            st.error(response)

    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": response,
        "sources": sources,
    })
    st.rerun()