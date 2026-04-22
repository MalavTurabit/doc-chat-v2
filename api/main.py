import tempfile
import os
import time
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response , FileResponse

from api.schemas import (
    UploadResponse, ChatRequest, ChatResponse,
    SessionInfo, DeleteResponse,
)
from api.session import (
    new_session_id,
    add_doc,
    get_docs,
    get_full_texts,
    get_memory,
    add_message,
    add_edit,
    get_edits,
    clear_session,
    session_info,
    save_image_path,
    get_image_path,
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
    session_id:          str,
    image_understanding: bool = False,
    file: UploadFile = File(...),
):
    suffix    = Path(file.filename).suffix.lower()
    supported = {
        ".pdf", ".docx", ".pptx", ".xlsx",
        ".csv", ".txt", ".png", ".jpg", ".jpeg"
    }

    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}.",
        )

    t0      = time.time()
    logger.info(f"[upload] started — {file.filename}  image_understanding={image_understanding}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content  = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    logger.info(f"[upload] file saved — {time.time()-t0:.2f}s  size={len(content)/1024:.1f}KB")

    try:
        t1  = time.time()
        doc = extract(tmp_path, image_understanding=image_understanding)
        doc["filename"] = file.filename
        logger.info(f"[upload] extracted {len(doc['blocks'])} blocks — {time.time()-t1:.2f}s")

        t2     = time.time()
        chunks = chunk_document(doc)
        logger.info(f"[upload] chunked into {len(chunks)} chunks — {time.time()-t2:.2f}s")

        # ── save extracted images to disk ─────────────────────────────────
                # ── save images to disk and store path on chunk ───────────────────
        image_map = doc.get("image_map", {})
        if image_map:
            image_dir = f"./extracted_images/{session_id}"
            os.makedirs(image_dir, exist_ok=True)

            for block_idx, image_bytes in image_map.items():
                block = doc["blocks"][block_idx]

                # find the chunk whose start_char matches this block
                target_chunk = next(
                    (c for c in chunks
                     if c["start_char"] == block["start_char"]),
                    None
                )

                if target_chunk:
                    chunk_id   = target_chunk["chunk_id"]
                    image_path = f"{image_dir}/{chunk_id}.png"

                    with open(image_path, "wb") as f_img:
                        f_img.write(image_bytes)

                    # store path directly on chunk — persisted to Milvus
                    target_chunk["image_path"] = image_path
                    target_chunk["has_image"]  = True

                    logger.info(f"[upload] image saved → {image_path}")

        t3     = time.time()
        chunks = embed_chunks(chunks)
        logger.info(f"[upload] embedded {len(chunks)} chunks — {time.time()-t3:.2f}s")

        t4 = time.time()
        for c in chunks:
            c["session_id"] = session_id
            c["filename"]   = file.filename
            if "has_image" not in c:
                c["has_image"] = False

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
        logger.warning(f"[upload] rejected — {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"[upload] unexpected error — {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error during upload.")

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
    docs = get_docs(req.session_id)

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
        image_refs=result.get("image_refs", []),
    )


# ── Download ──────────────────────────────────────────────────────────────────

@app.get("/download/{session_id}")
def download(session_id: str, doc_id: str = None):
    from vectorstore.milvus_client import get_all_chunks

    if doc_id:
        # try memory first
        doc      = get_doc(session_id, doc_id) if hasattr(get_docs, '__self__') else None
        docs     = get_docs(session_id)
        doc      = next((d for d in docs if d["doc_id"] == doc_id), None)
        full_text = doc["full_text"] if doc and doc.get("full_text") else ""

        # if full_text empty (after restart) reconstruct from Milvus chunks
        if not full_text:
            chunks    = get_all_chunks(session_id=session_id, doc_id=doc_id)
            full_text = "\n\n".join(c["text"] for c in sorted(
                chunks, key=lambda x: x.get("start_char", 0)
            ))

        filename = doc["filename"] if doc else f"{doc_id}.txt"
        edits    = get_edits(session_id)
        doc_edits = [e for e in edits if e.get("doc_id") == doc_id]
        content  = reconstruct_as_txt(full_text, doc_edits)

        return Response(
            content=content,
            media_type="text/plain",
            headers={
                "Content-Disposition":
                    f'attachment; filename="{Path(filename).stem}_updated.txt"'
            },
        )
    else:
        # all docs combined
        docs      = get_docs(session_id)
        all_texts = []

        for d in docs:
            full_text = d.get("full_text", "")
            if not full_text:
                chunks    = get_all_chunks(session_id=session_id, doc_id=d["doc_id"])
                full_text = "\n\n".join(c["text"] for c in sorted(
                    chunks, key=lambda x: x.get("start_char", 0)
                ))
            edits   = [e for e in get_edits(session_id) if e.get("doc_id") == d["doc_id"]]
            content = reconstruct_as_txt(full_text, edits)
            all_texts.append(f"=== {d['filename']} ===\n{content}")

        combined = "\n\n".join(all_texts)
        return Response(
            content=combined,
            media_type="text/plain",
            headers={
                "Content-Disposition":
                    'attachment; filename="all_documents_updated.txt"'
            },
        )


# ── Delete session ────────────────────────────────────────────────────────────

@app.delete("/session/{session_id}", response_model=DeleteResponse)
def delete_session_route(session_id: str):
    delete_session(session_id)
    clear_session(session_id)
    logger.info(f"[session] deleted — {session_id}")
    return DeleteResponse(session_id=session_id, deleted=True)

#-- Serve extracted images ─────────────────────────────────────────────────────
from fastapi.responses import FileResponse

@app.get("/image/{chunk_id}")
def get_image(chunk_id: str):
    """Serve image by looking up its stored path from Milvus."""
    from vectorstore.milvus_client import get_client
    from config import MILVUS_COLLECTION

    client = get_client()

    try:
        results = client.query(
            collection_name=MILVUS_COLLECTION,
            filter=f'chunk_id == "{chunk_id}"',
            output_fields=["image_path"],
            limit=1,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus query failed: {e}")

    if not results or not results[0].get("image_path"):
        raise HTTPException(
            status_code=404,
            detail=f"No image registered for chunk '{chunk_id}'."
        )

    image_path = results[0]["image_path"]

    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail=f"Image file missing from disk: {image_path}"
        )

    return FileResponse(image_path, media_type="image/png")

@app.get("/image_b64/{chunk_id}")
def get_image_b64(chunk_id: str):
    """Return image as base64 string for embedding in Streamlit."""
    import base64
    from vectorstore.milvus_client import get_client
    from config import MILVUS_COLLECTION

    client = get_client()
    try:
        results = client.query(
            collection_name=MILVUS_COLLECTION,
            filter=f'chunk_id == "{chunk_id}"',
            output_fields=["image_path"],
            limit=1,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not results or not results[0].get("image_path"):
        raise HTTPException(status_code=404, detail="No image for this chunk.")

    image_path = results[0]["image_path"]
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file missing.")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return {"chunk_id": chunk_id, "b64": b64, "media_type": "image/png"}