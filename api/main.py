import tempfile
import os
import time
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Doc Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_collection()
    logger.info("Milvus collection ready.")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Session ───────────────────────────────────────────────────────────────────

@app.post("/session")
def create_session() -> dict:
    sid = new_session_id()
    logger.info(f"[session] created — {sid}")
    return {"session_id": sid}


@app.get("/session/{session_id}", response_model=SessionInfo)
def get_session(session_id: str):
    return session_info(session_id)


# ── Upload ────────────────────────────────────────────────────────────────────

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

    t0 = time.time()
    logger.info(f"[upload] started — {file.filename}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content  = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    logger.info(
        f"[upload] file saved — {time.time()-t0:.2f}s  "
        f"size={len(content)/1024:.1f}KB"
    )

    try:
        t1 = time.time()
        doc = extract(tmp_path)
        doc["filename"] = file.filename
        logger.info(
            f"[upload] extracted {len(doc['blocks'])} blocks — "
            f"{time.time()-t1:.2f}s"
        )

        t2 = time.time()
        chunks = chunk_document(doc)
        logger.info(
            f"[upload] chunked into {len(chunks)} chunks — "
            f"{time.time()-t2:.2f}s"
        )

        t3 = time.time()
        chunks = embed_chunks(chunks)
        logger.info(
            f"[upload] embedded {len(chunks)} chunks — "
            f"{time.time()-t3:.2f}s"
        )

        t4 = time.time()
        for c in chunks:
            c["session_id"] = session_id
            c["filename"]   = file.filename

        upsert_chunks(chunks)
        logger.info(f"[upload] upserted to Milvus — {time.time()-t4:.2f}s")

        add_doc(
            session_id=session_id,
            doc_id=doc["doc_id"],
            filename=file.filename,
            full_text=doc["full_text"],
        )

        logger.info(f"[upload] total — {time.time()-t0:.2f}s")

    except ValueError as e:
        # clean user-facing errors — image-based PDF, unsupported content, etc.
        logger.warning(f"[upload] rejected — {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # unexpected errors — log full details server side
        logger.error(f"[upload] unexpected error — {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during upload. Check server logs.",
        )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return UploadResponse(
        doc_id=doc["doc_id"],
        filename=file.filename,
        session_id=session_id,
        chunks=len(chunks),
    )


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not get_docs(req.session_id):
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded for this session.",
        )

    t0 = time.time()
    logger.info(f"[chat] query='{req.query[:60]}...'  session={req.session_id[:8]}")

    memory = get_memory(req.session_id)

    result = run(
        query=req.query,
        session_id=req.session_id,
        memory=memory,
    )

    if result.get("edit_record"):
        add_edit(req.session_id, result["edit_record"])

    add_message(req.session_id, "user",      req.query)
    add_message(req.session_id, "assistant", result["response"])

    logger.info(
        f"[chat] intent={result['intent']}  "
        f"sources={result['sources']}  "
        f"total={time.time()-t0:.2f}s"
    )

    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        sources=result["sources"],
    )


# ── Download ──────────────────────────────────────────────────────────────────

@app.get("/download/{session_id}")
def download(session_id: str, doc_id: str | None = None):
    full_texts = get_full_texts(session_id)
    edits      = get_edits(session_id)

    if not full_texts:
        raise HTTPException(status_code=404, detail="No documents found.")

    if doc_id:
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
        parts = []
        for d in get_docs(session_id):
            did       = d["doc_id"]
            doc_edits = [e for e in edits if e["doc_id"] == did]
            updated   = reconstruct_as_txt(full_texts[did], doc_edits)
            parts.append(f"=== {d['filename']} ===\n\n{updated}")
        text = "\n\n\n".join(parts)
        stem = "all_documents"

    logger.info(f"[download] session={session_id[:8]}  doc_id={doc_id}  chars={len(text)}")

    from fastapi.responses import Response
    return Response(
        content=text.encode("utf-8"),
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{stem}_updated.txt"'
        },
    )


# ── Delete session ────────────────────────────────────────────────────────────

@app.delete("/session/{session_id}", response_model=DeleteResponse)
def delete_session_route(session_id: str):
    delete_session(session_id)
    clear_session(session_id)
    logger.info(f"[session] deleted — {session_id}")
    return DeleteResponse(session_id=session_id, deleted=True)