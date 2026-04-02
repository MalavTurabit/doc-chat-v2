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
    doc = index_document(path)
    doc_id   = doc["doc_id"]
    filename = doc["filename"]
    print(f"doc_id: {doc_id}")

    queries = [
        ("qa",        "What is the leave policy?"),
        ("summarise", "Summarise the document"),
        ("explain",   "Explain the code of conduct in simple terms"),
    ]

    for label, query in queries:
        print(f"\n{'='*60}")
        print(f"Query [{label}]: {query}")
        print(f"{'='*60}")
        response = run(query, doc_id, filename)
        print(response)