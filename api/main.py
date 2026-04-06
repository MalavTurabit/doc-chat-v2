import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    UploadResponse, ChatRequest, ChatResponse,
    SessionInfo, DeleteResponse,
)
from api.session import (
    new_session_id, add_doc, get_docs, get_doc_ids,
    get_full_texts, add_message, get_memory,
    add_edit, get_edits, clear_session, session_info,
)
from ingestion.parser import extract
from vectorstore.chunker import chunk_document
from vectorstore.embedder import embed_chunks
from vectorstore.milvus_client import init_collection, upsert_chunks, delete_session
from graph.graph import run
from export.reconstructor import reconstruct_as_txt

app = FastAPI(title="Doc Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# init Milvus on startup
@app.on_event("startup")
def startup():
    init_collection()


# ── POST /session — create new session ───────────────────────────────────────

@app.post("/session")
def create_session() -> dict:
    sid = new_session_id()
    return {"session_id": sid}


# ── GET /session/{session_id} — session info ──────────────────────────────────

@app.get("/session/{session_id}", response_model=SessionInfo)
def get_session(session_id: str):
    return session_info(session_id)


# ── POST /upload — upload and index a document ────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    session_id: str,
    file: UploadFile = File(...),
):
    suffix = Path(file.filename).suffix.lower()
    supported = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt"}

    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {supported}",
        )

    # save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content  = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc    = extract(tmp_path)
        doc["filename"] = file.filename
        chunks = chunk_document(doc)
        chunks = embed_chunks(chunks)

        # attach session_id and correct filename to every chunk
        for c in chunks:
            c["session_id"] = session_id
            c["filename"]   = file.filename   # ensure chunks carry real name too

        upsert_chunks(chunks)

        add_doc(
            session_id=session_id,
            doc_id=doc["doc_id"],
            filename=file.filename,
            full_text=doc["full_text"],
        )

    finally:
        os.unlink(tmp_path)

    return UploadResponse(
        doc_id=doc["doc_id"],
        filename=file.filename,
        session_id=session_id,
        chunks=len(chunks),
    )


# ── POST /chat — unified multi-doc chat ───────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not get_docs(req.session_id):
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded for this session.",
        )

    memory = get_memory(req.session_id)

    result = run(
        query=req.query,
        session_id=req.session_id,
        memory=memory,
    )

    # store edit record if this was an edit
    if result.get("edit_record"):
        add_edit(req.session_id, result["edit_record"])

    # update memory
    add_message(req.session_id, "user",      req.query)
    add_message(req.session_id, "assistant", result["response"])

    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        sources=result["sources"],
    )


# ── GET /download/{session_id} — download updated txt ────────────────────────

@app.get("/download/{session_id}")
def download(session_id: str, doc_id: str | None = None):
    full_texts = get_full_texts(session_id)
    edits      = get_edits(session_id)

    if not full_texts:
        raise HTTPException(status_code=404, detail="No documents found.")

    if doc_id:
        # download single doc
        if doc_id not in full_texts:
            raise HTTPException(status_code=404, detail="doc_id not found.")
        doc_edits = [e for e in edits if e["doc_id"] == doc_id]
        text      = reconstruct_as_txt(full_texts[doc_id], doc_edits)
        docs      = get_docs(session_id)
        filename  = next(
            (d["filename"] for d in docs if d["doc_id"] == doc_id), "document"
        )
        stem = Path(filename).stem
    else:
        # download all docs combined
        parts = []
        for d in get_docs(session_id):
            did      = d["doc_id"]
            doc_edits = [e for e in edits if e["doc_id"] == did]
            updated  = reconstruct_as_txt(full_texts[did], doc_edits)
            parts.append(f"=== {d['filename']} ===\n\n{updated}")
        text = "\n\n\n".join(parts)
        stem = "all_documents"

    from fastapi.responses import Response
    return Response(
        content=text.encode("utf-8"),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{stem}_updated.txt"'},
    )


# ── DELETE /session/{session_id} ──────────────────────────────────────────────

@app.delete("/session/{session_id}", response_model=DeleteResponse)
def delete_session_route(session_id: str):
    delete_session(session_id)
    clear_session(session_id)
    return DeleteResponse(session_id=session_id, deleted=True)