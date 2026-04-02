import sys
from ingestion.parser import extract
from vectorstore.chunker import chunk_document
from vectorstore.embedder import embed_chunks
from vectorstore.milvus_client import init_collection, upsert_chunks, get_all_chunks
from graph.graph import run


def index_document(path: str) -> dict:
    doc    = extract(path)
    chunks = chunk_document(doc)
    chunks = embed_chunks(chunks)
    init_collection()
    upsert_chunks(chunks)
    return doc


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/malavj/Downloads/Employee Handbook.pdf"

    print("\n--- Indexing document ---")
    doc      = index_document(path)
    doc_id   = doc["doc_id"]
    filename = doc["filename"]
    print(f"doc_id   : {doc_id}")
    print(f"filename : {filename}")

    # ── QA ────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Query [qa]: What is the leave policy?")
    print(f"{'='*60}")
    result = run(
        query="What is the leave policy?",
        doc_id=doc_id,
        filename=filename,
    )
    print(result["response"])

    # ── SUMMARISE ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Query [summarise]: Summarise the document")
    print(f"{'='*60}")
    result = run(
        query="Summarise the document",
        doc_id=doc_id,
        filename=filename,
    )
    print(result["response"])

    # ── EXPLAIN ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Query [explain]: Explain the code of conduct in simple terms")
    print(f"{'='*60}")
    result = run(
        query="Explain the code of conduct in simple terms",
        doc_id=doc_id,
        filename=filename,
    )
    print(result["response"])

    # ── EDIT ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Query [edit]: Change annual leave from 18 days to 25 days")
    print(f"{'='*60}")
    result = run(
        query="Change annual leave from 18 days to 25 days",
        doc_id=doc_id,
        filename=filename,
    )
    print(result["response"])
    print(f"\nEdit history entries : {len(result['edit_history'])}")
    if result["edit_history"]:
        last = result["edit_history"][-1]
        print(f"Section edited       : {last['section']}")
        print(f"Instruction          : {last['instruction']}")
        print(f"Original preview     : {last['original_text'][:120]}...")
        print(f"Edited preview       : {last['edited_text'][:120]}...")

    # ── SECOND EDIT — verify history accumulates ───────────────────────────────
    print(f"\n{'='*60}")
    print("Query [edit]: Make the code of conduct more formal in tone")
    print(f"{'='*60}")
    result = run(
        query="Make the code of conduct more formal in tone",
        doc_id=doc_id,
        filename=filename,
        edit_history=result["edit_history"],   # pass history forward
    )
    print(result["response"])
    print(f"\nEdit history entries : {len(result['edit_history'])}")
    for i, entry in enumerate(result["edit_history"], 1):
        print(f"  [{i}] section={entry['section']}  instruction={entry['instruction']}")