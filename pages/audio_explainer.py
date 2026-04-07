import streamlit as st
import tempfile
import os
from pathlib import Path
import requests

from config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Audio Explainer",
    page_icon="🎙️",
    layout="centered",
)

# ── Session state ─────────────────────────────────────────────────────────────

def init():
    defaults = {
        "audio_session_id": None,
        "audio_filename":   None,
        "audio_script":     None,
        "audio_bytes":      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_session() -> str:
    res = requests.post(f"{API_BASE}/session")
    return res.json()["session_id"]


def upload_doc(session_id: str, uploaded_file) -> dict:
    res = requests.post(
        f"{API_BASE}/upload",
        params={"session_id": session_id},
        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
    )
    return res.json()


def generate_script(session_id: str) -> str:
    res = requests.post(
        f"{API_BASE}/chat",
        json={
            "session_id": session_id,
            "query": (
            "You are my best friend explaining this document to me the night before my exam. "
            "I am stressed and need to understand it quickly. "
            "Talk to me like a friend — casual, warm, encouraging. "
            "Use simple words, real examples, and analogies where it helps. "
            "Occasionally say things like 'okay so listen', 'this is the important bit', "
            "'don't worry this part is simple', 'trust me you need to remember this one'. "
            "Cover all the key points, important numbers, rules and policies. "
            "No bullet points, no markdown, no headings — just talk to me naturally like "
            "we are sitting together the night before the exam. "
            "Keep it under 900 words and make sure I actually feel ready after hearing it."
            ),
        },
    )
    return res.json()["response"]


def text_to_speech(script: str) -> bytes:
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio  = client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=script,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    return b"".join(chunk for chunk in audio if chunk)


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🎙️ Audio explainer")
st.caption("Upload a document and get a spoken explanation of it.")
st.divider()

uploaded = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "pptx", "xlsx", "csv", "txt"],
    key="audio_uploader",
)

if uploaded:
    if uploaded.name != st.session_state.audio_filename:
        # clear previous state
        if st.session_state.audio_session_id:
            requests.delete(f"{API_BASE}/session/{st.session_state.audio_session_id}")

        st.session_state.audio_session_id = None
        st.session_state.audio_filename   = None
        st.session_state.audio_script     = None
        st.session_state.audio_bytes      = None

        with st.spinner(f"Indexing {uploaded.name}..."):
            session_id = create_session()
            upload_doc(session_id, uploaded)

        st.session_state.audio_session_id = session_id
        st.session_state.audio_filename   = uploaded.name
        st.success(f"Ready — {uploaded.name}")

# ── Generate ──────────────────────────────────────────────────────────────────

if st.session_state.audio_filename:
    st.divider()
    st.markdown(f"**Document:** `{st.session_state.audio_filename}`")

    col1, col2 = st.columns(2)

    with col1:
        generate_clicked = st.button(
            "Generate audio explanation",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.audio_bytes is not None,
        )

    with col2:
        if st.session_state.audio_bytes:
            if st.button("Regenerate", use_container_width=True):
                st.session_state.audio_script = None
                st.session_state.audio_bytes  = None
                st.rerun()

    if generate_clicked:
        with st.spinner("Writing script..."):
            script = generate_script(st.session_state.audio_session_id)
        st.session_state.audio_script = script

        with st.spinner("Generating audio with ElevenLabs..."):
            audio_bytes = text_to_speech(script)
        st.session_state.audio_bytes = audio_bytes

        st.rerun()

# ── Playback + download ───────────────────────────────────────────────────────

if st.session_state.audio_bytes:
    st.divider()
    st.markdown("**Listen**")
    st.audio(st.session_state.audio_bytes, format="audio/mp3")

    stem = Path(st.session_state.audio_filename).stem
    st.download_button(
        label="Download MP3",
        data=st.session_state.audio_bytes,
        file_name=f"{stem}_explanation.mp3",
        mime="audio/mpeg",
        use_container_width=True,
    )

# ── Script preview ────────────────────────────────────────────────────────────

if st.session_state.audio_script:
    st.divider()
    with st.expander("View script"):
        st.write(st.session_state.audio_script)