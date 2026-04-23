import uuid
import os
import shutil
from collections import defaultdict


# ── In-memory stores (per process) ───────────────────────────────────────────

# session_id → list of {doc_id, filename, full_text}
_docs: dict[str, list[dict]] = defaultdict(list)

# session_id → list of {role, content}  (last N messages)
_memory: dict[str, list[dict]] = defaultdict(list)

# session_id → list of edit records
_edits: dict[str, list[dict]] = defaultdict(list)

# chunk_id → image file path
_images: dict[str, str] = {}

MEMORY_WINDOW = 6   # number of messages to keep in context


# ── Session ID ────────────────────────────────────────────────────────────────

def new_session_id() -> str:
    return str(uuid.uuid4())


# ── Docs ──────────────────────────────────────────────────────────────────────

def add_doc(session_id: str, doc_id: str, filename: str, full_text: str):
    _docs[session_id].append({
        "doc_id":    doc_id,
        "filename":  filename,
        "full_text": full_text,
    })


def get_docs(session_id: str) -> list[dict]:
    # return from memory if available
    if _docs.get(session_id):
        return _docs[session_id]

    # memory is empty (FastAPI restarted) — rebuild from Milvus
    try:
        from vectorstore.milvus_client import get_all_chunks
        chunks = get_all_chunks(session_id=session_id)
        if not chunks:
            return []

        # group chunks by doc_id to reconstruct doc list
        docs_seen = {}
        for c in chunks:
            doc_id   = c["doc_id"]
            filename = c["filename"]
            if doc_id not in docs_seen:
                docs_seen[doc_id] = {
                    "doc_id":    doc_id,
                    "filename":  filename,
                    "full_text": "",   # full_text not stored in Milvus
                }

        reconstructed = list(docs_seen.values())
        _docs[session_id] = reconstructed
        print(f"[session] rebuilt {len(reconstructed)} docs from Milvus for session {session_id}")
        return reconstructed

    except Exception as e:
        print(f"[session] failed to rebuild from Milvus: {e}")
        return []


def get_doc_ids(session_id: str) -> list[str]:
    return [d["doc_id"] for d in get_docs(session_id)]


def get_full_texts(session_id: str) -> dict[str, str]:
    """Returns {doc_id: full_text} for all docs in session."""
    return {d["doc_id"]: d["full_text"] for d in get_docs(session_id)}


# ── Memory ────────────────────────────────────────────────────────────────────

def add_message(session_id: str, role: str, content: str):
    _memory[session_id].append({"role": role, "content": content})
    if len(_memory[session_id]) > MEMORY_WINDOW:
        _memory[session_id] = _memory[session_id][-MEMORY_WINDOW:]


def get_memory(session_id: str) -> list[dict]:
    return _memory.get(session_id, [])


# ── Edit history ──────────────────────────────────────────────────────────────

def add_edit(session_id: str, edit_record: dict):
    _edits[session_id].append(edit_record)


def get_edits(session_id: str) -> list[dict]:
    return _edits.get(session_id, [])


# ── Images ────────────────────────────────────────────────────────────────────

def save_image_path(chunk_id: str, image_path: str):
    """Register a chunk_id → image file path mapping."""
    _images[chunk_id] = image_path


def get_image_path(chunk_id: str) -> str | None:
    """Get the image file path for a chunk_id."""
    return _images.get(chunk_id)


def has_image(chunk_id: str) -> bool:
    """Check if a chunk has an associated image."""
    return chunk_id in _images


# ── Clear session ─────────────────────────────────────────────────────────────

def clear_session(session_id: str):
    _docs.pop(session_id,   None)
    _memory.pop(session_id, None)
    _edits.pop(session_id,  None)

    # delete extracted images for this session from disk
    image_dir = f"./extracted_images/{session_id}"
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
        print(f"[session] deleted images for session {session_id}")

    # remove chunk_id entries that belonged to this session
    # (they are keyed by chunk_id so we can't filter by session directly —
    #  image files are already deleted above so orphaned keys are harmless)


# ── Session info ──────────────────────────────────────────────────────────────

def session_info(session_id: str) -> dict:
    return {
        "session_id":    session_id,
        "docs":          [
            {"doc_id": d["doc_id"], "filename": d["filename"]}
            for d in get_docs(session_id)
        ],
        "message_count": len(get_memory(session_id)),
        "edit_count":    len(get_edits(session_id)),
    }