import uuid
from collections import defaultdict


# ── In-memory stores (per process) ───────────────────────────────────────────

# session_id → list of {doc_id, filename, full_text}
_docs: dict[str, list[dict]] = defaultdict(list)

# session_id → list of {role, content}  (last N messages)
_memory: dict[str, list[dict]] = defaultdict(list)

# session_id → list of edit records
_edits: dict[str, list[dict]] = defaultdict(list)

MEMORY_WINDOW = 6   # number of messages to keep in context


# ── Session ID ────────────────────────────────────────────────────────────────

def new_session_id() -> str:
    return str(uuid.uuid4())


# ── Docs ─────────────────────────────────────────────────────────────────────

def add_doc(session_id: str, doc_id: str, filename: str, full_text: str):
    _docs[session_id].append({
        "doc_id":    doc_id,
        "filename":  filename,
        "full_text": full_text,
    })


def get_docs(session_id: str) -> list[dict]:
    return _docs.get(session_id, [])


def get_doc_ids(session_id: str) -> list[str]:
    return [d["doc_id"] for d in get_docs(session_id)]


def get_full_texts(session_id: str) -> dict[str, str]:
    """Returns {doc_id: full_text} for all docs in session."""
    return {d["doc_id"]: d["full_text"] for d in get_docs(session_id)}


# ── Memory ────────────────────────────────────────────────────────────────────

def add_message(session_id: str, role: str, content: str):
    _memory[session_id].append({"role": role, "content": content})
    # keep only last MEMORY_WINDOW messages
    if len(_memory[session_id]) > MEMORY_WINDOW:
        _memory[session_id] = _memory[session_id][-MEMORY_WINDOW:]


def get_memory(session_id: str) -> list[dict]:
    return _memory.get(session_id, [])


# ── Edit history ──────────────────────────────────────────────────────────────

def add_edit(session_id: str, edit_record: dict):
    _edits[session_id].append(edit_record)


def get_edits(session_id: str) -> list[dict]:
    return _edits.get(session_id, [])


# ── Clear session ─────────────────────────────────────────────────────────────

def clear_session(session_id: str):
    _docs.pop(session_id, None)
    _memory.pop(session_id, None)
    _edits.pop(session_id, None)


# ── Session info ──────────────────────────────────────────────────────────────

def session_info(session_id: str) -> dict:
    return {
        "session_id":    session_id,
        "docs":          [{"doc_id": d["doc_id"], "filename": d["filename"]}
                          for d in get_docs(session_id)],
        "message_count": len(get_memory(session_id)),
        "edit_count":    len(get_edits(session_id)),
    }