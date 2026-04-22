import streamlit as st
import requests
import base64
from pathlib import Path

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Doc Chat",
    page_icon="📄",
    layout="wide",
)

def init_session():
    if "session_id" not in st.session_state:
        try:
            res = requests.post(f"{API_BASE}/session", timeout=5)
            st.session_state.session_id = res.json()["session_id"]
        except Exception:
            st.error("Cannot reach the FastAPI backend. Make sure it is running on port 8000.")
            st.stop()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "docs" not in st.session_state:
        st.session_state.docs = []

init_session()


def display_images(image_refs: list[str]):
    if not image_refs:
        return
    st.markdown("**Referenced images** (click to view):")
    for i, chunk_id in enumerate(image_refs):
        url = f"{API_BASE}/image/{chunk_id}"
        st.markdown(
            f'🖼️ <a href="{url}" target="_blank">View image {i+1}</a>',
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.title("📄 Doc Chat")
    st.caption("Upload documents and chat with all of them.")
    st.page_link("pages/audio_explainer.py", label="🎙️ Audio explainer")
    st.divider()

    image_understanding = st.checkbox(
        "🔍 Enable image understanding",
        value=False,
        help=(
            "Uses GPT-4.1-mini vision to describe charts, diagrams, "
            "and embedded images inside documents. "
            "Adds processing time and token cost per image."
        ),
    )

    if image_understanding and st.session_state.docs:
        st.warning(
            "Documents already uploaded without image understanding. "
            "Clear and re-upload with the toggle enabled to extract images.",
            icon="⚠️",
        )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "pptx", "xlsx", "csv", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, PPTX, XLSX, CSV, TXT, PNG, JPG",
    )

    if uploaded_files:
        existing_names = {d["filename"] for d in st.session_state.docs}
        new_files = [f for f in uploaded_files if f.name not in existing_names]

        for uploaded in new_files:
            progress = st.progress(0, text=f"Uploading {uploaded.name}...")
            try:
                res = requests.post(
                    f"{API_BASE}/upload",
                    params={
                        "session_id":          st.session_state.session_id,
                        "image_understanding": str(image_understanding).lower(),
                    },
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=None,
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
                        f"Ready — {uploaded.name} ({data['chunks']} chunks)"
                        + (" 🔍" if image_understanding else "")
                    )
                elif res.status_code == 400:
                    st.error(res.json().get("detail", res.text))
                else:
                    st.error(f"Upload failed ({res.status_code}): {res.text}")

            except requests.exceptions.Timeout:
                st.error(f"Upload timed out for {uploaded.name}.")
            except Exception as e:
                st.error(f"Upload error: {e}")

    if st.session_state.docs:
        st.divider()
        st.markdown("**Uploaded documents**")
        for d in st.session_state.docs:
            st.caption(f"📄 {d['filename']}  ({d['chunks']} chunks)")

    if st.session_state.docs:
        st.divider()
        st.markdown("**Download updated TXT**")

        if len(st.session_state.docs) == 1:
            d   = st.session_state.docs[0]
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


if not st.session_state.docs:
    st.markdown("## Welcome to Doc Chat")
    st.markdown(
        "Upload one or more **PDF, DOCX, PPTX, XLSX, CSV, TXT, PNG, or JPG** "
        "files from the sidebar to get started.\n\n"
        "You can:\n"
        "- Ask questions across all uploaded documents\n"
        "- The bot will tell you which document the answer came from\n"
        "- Request summaries or simple explanations\n"
        "- Edit sections with natural language instructions\n"
        "- Download the updated document as plain text\n"
        "- Get an audio explanation via the 🎙️ Audio Explainer\n"
        "- Enable 🔍 image understanding to extract charts and diagrams\n"
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
        if msg.get("image_refs"):
            display_images(msg["image_refs"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")
        if msg.get("intent"):
            st.caption(f"Intent: {msg['intent']}")

# ── Chat input ────────────────────────────────────────────────────────────────

query = st.chat_input("Ask anything across your documents...")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})

    response   = ""
    sources    = []
    intent     = ""
    image_refs = []

    with st.spinner("Thinking..."):
        try:
            res = requests.post(
                f"{API_BASE}/chat",
                json={
                    "session_id": st.session_state.session_id,
                    "query":      query,
                },
                timeout=None,
            )

            if res.status_code == 200:
                data       = res.json()
                response   = data["response"]
                sources    = data.get("sources", [])
                intent     = data.get("intent", "")
                image_refs = data.get("image_refs", [])
            else:
                response = f"Error {res.status_code}: {res.text}"

        except requests.exceptions.Timeout:
            response = "Request timed out. Try a shorter query."

        except Exception as e:
            response = f"Error: {e}"

    st.session_state.chat_history.append({
        "role":       "assistant",
        "content":    response,
        "sources":    sources,
        "intent":     intent,
        "image_refs": image_refs,
    })
    st.rerun()