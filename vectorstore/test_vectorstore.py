import sys
from ingestion.parser import extract
from vectorstore.chunker import chunk_document
from vectorstore.embedder import embed_chunks, embed_query
from vectorstore.milvus_client import init_collection, upsert_chunks, search, get_all_chunks


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/malavj/Downloads/Employee Handbook.pdf"

    # 1. init collection
    print("\n--- Step 1: init Milvus collection ---")
    init_collection()

    # 2. extract
    print("\n--- Step 2: extract text ---")
    doc = extract(path)
    print(f"Extracted {len(doc['blocks'])} blocks from {doc['filename']}")

    # 3. chunk
    print("\n--- Step 3: chunk ---")
    chunks = chunk_document(doc)
    print(f"Created {len(chunks)} chunks")

    # 4. embed
    print("\n--- Step 4: embed ---")
    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks — dim={len(chunks[0]['embedding'])}")

    # 5. upsert
    print("\n--- Step 5: upsert into Milvus ---")
    upsert_chunks(chunks)

    # 6. verify all chunks stored
    print("\n--- Step 6: verify stored chunks ---")
    stored = get_all_chunks(doc["doc_id"])
    print(f"Chunks in Milvus for this doc: {len(stored)}")

    # 7. search
    print("\n--- Step 7: search ---")
    query = "What is the leave policy?"
    q_vec = embed_query(query)
    hits  = search(q_vec, doc["doc_id"], top_k=3)

    print(f"Query: '{query}'")
    print(f"Top {len(hits)} results:\n")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] score={h['score']:.4f}  section='{h['section_heading']}'  page={h['page']}")
        print(f"       {h['text'][:100]}...")
        print()