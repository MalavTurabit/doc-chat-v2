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
        try:
            res = requests.post(f"{API_BASE}/session", timeout=5)
            st.session_state.session_id = res.json()["session_id"]
        except Exception:
            st.error(
                "Cannot reach the FastAPI backend. "
                "Make sure it is running on port 8000."
            )
            st.stop()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "docs" not in st.session_state:
        st.session_state.docs = []

init_session()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 Doc Chat")
    st.caption("Upload documents and chat with all of them.")
    st.page_link("pages/audio_explainer.py", label="🎙️ Audio explainer")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "pptx", "xlsx", "csv", "txt"],
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, PPTX, XLSX, CSV, TXT",
    )

    if uploaded_files:
        existing_names = {d["filename"] for d in st.session_state.docs}
        new_files = [f for f in uploaded_files if f.name not in existing_names]

        for uploaded in new_files:
            progress = st.progress(0, text=f"Uploading {uploaded.name}...")

            try:
                res = requests.post(
                    f"{API_BASE}/upload",
                    params={"session_id": st.session_state.session_id},
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout= None,  # no timeout for uploads — can take a while for large files
                )
                progress.progress(100, text=f"Done — {uploaded.name}")

                if res.status_code == 200:
                    data = res.json()
                    st.session_state.docs.append({
                        "doc_id":   data["doc_id"],
                        "filename": data["filename"],
                        "chunks":   data["chunks"],
                    })
                    st.success(
                        f"Ready — {uploaded.name} "
                        f"({data['chunks']} chunks)"
                    )
                elif res.status_code == 400:
                    # clean user-facing error from our guards
                    detail = res.json().get("detail", res.text)
                    st.error(f"{detail}")
                else:
                    st.error(f"Upload failed ({res.status_code}): {res.text}")

            except requests.exceptions.Timeout:
                st.error(
                    f"Upload timed out for {uploaded.name}. "
                    "File may be too large or backend is slow."
                )
            except Exception as e:
                st.error(f"Upload error: {e}")

    # ── Uploaded docs list ────────────────────────────────────────────────────

    if st.session_state.docs:
        st.divider()
        st.markdown("**Uploaded documents**")
        for d in st.session_state.docs:
            st.caption(f"📄 {d['filename']}  ({d['chunks']} chunks)")

    # ── Downloads ─────────────────────────────────────────────────────────────

    if st.session_state.docs:
        st.divider()
        st.markdown("**Download updated TXT**")

        if len(st.session_state.docs) == 1:
            d    = st.session_state.docs[0]
            res  = requests.get(
                f"{API_BASE}/download/{st.session_state.session_id}",
                params={"doc_id": d["doc_id"]},
                timeout=30,
            )
            st.download_button(
                label=f"{Path(d['filename']).stem}_updated.txt",
                data=res.content,
                file_name=f"{Path(d['filename']).stem}_updated.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            for d in st.session_state.docs:
                res = requests.get(
                    f"{API_BASE}/download/{st.session_state.session_id}",
                    params={"doc_id": d["doc_id"]},
                    timeout=30,
                )
                st.download_button(
                    label=f"{Path(d['filename']).stem}_updated.txt",
                    data=res.content,
                    file_name=f"{Path(d['filename']).stem}_updated.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=d["doc_id"],
                )

            res = requests.get(
                f"{API_BASE}/download/{st.session_state.session_id}",
                timeout=30,
            )
            st.download_button(
                label="Download all as single TXT",
                data=res.content,
                file_name="all_documents_updated.txt",
                mime="text/plain",
                use_container_width=True,
                key="all",
            )

    # ── Clear ─────────────────────────────────────────────────────────────────

    st.divider()
    if st.button("Clear & start over", use_container_width=True):
        try:
            requests.delete(
                f"{API_BASE}/session/{st.session_state.session_id}",
                timeout=10,
            )
        except Exception:
            pass
        for key in ["session_id", "chat_history", "docs"]:
            del st.session_state[key]
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────

if not st.session_state.docs:
    st.markdown("## Welcome to Doc Chat")
    st.markdown(
        "Upload one or more **PDF, DOCX, PPTX, XLSX, CSV, or TXT** files "
        "from the sidebar to get started.\n\n"
        "You can:\n"
        "- Ask questions across all uploaded documents\n"
        "- The bot will tell you which document the answer came from\n"
        "- Request summaries or simple explanations\n"
        "- Edit sections with natural language instructions\n"
        "- Download the updated document as plain text\n"
        "- Get an audio explanation via the 🎙️ Audio Explainer\n"
    )
    st.stop()

st.markdown("### Chat with your documents")
doc_names = ", ".join(d["filename"] for d in st.session_state.docs)
st.caption(f"Active: {doc_names}")
st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")
        if msg.get("intent"):
            st.caption(f"Intent: {msg['intent']}")

# ── Chat input ────────────────────────────────────────────────────────────────

query = st.chat_input("Ask anything across your documents...")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    f"{API_BASE}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "query":      query,
                    },
                    timeout= None,
                )

                if res.status_code == 200:
                    data     = res.json()
                    response = data["response"]
                    sources  = data.get("sources", [])
                    intent   = data.get("intent", "")
                    st.markdown(response)
                    if sources:
                        st.caption(f"Sources: {', '.join(sources)}")
                    if intent:
                        st.caption(f"Intent: {intent}")
                else:
                    response = f"Error {res.status_code}: {res.text}"
                    sources  = []
                    intent   = ""
                    st.error(response)

            except requests.exceptions.Timeout:
                response = "Request timed out. Try a shorter query."
                sources  = []
                intent   = ""
                st.error(response)

            except Exception as e:
                response = f"Error: {e}"
                sources  = []
                intent   = ""
                st.error(response)

    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": response,
        "sources": sources,
        "intent":  intent,
    })
    st.rerun()