from pymilvus import MilvusClient, DataType
from config import MILVUS_DB_PATH, MILVUS_COLLECTION, EMBEDDING_DIM

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(MILVUS_DB_PATH)
    return _client


def init_collection():
    client = get_client()

    if client.has_collection(MILVUS_COLLECTION):
        print(f"Collection '{MILVUS_COLLECTION}' already exists — skipping creation.")
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

    schema.add_field("chunk_id",        DataType.VARCHAR, max_length=200, is_primary=True)
    schema.add_field("embedding",       DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("doc_id",          DataType.VARCHAR, max_length=100)
    schema.add_field("session_id",      DataType.VARCHAR, max_length=100)
    schema.add_field("filename",        DataType.VARCHAR, max_length=300)
    schema.add_field("text",            DataType.VARCHAR, max_length=4000)
    schema.add_field("section_heading", DataType.VARCHAR, max_length=500)
    schema.add_field("page",            DataType.VARCHAR, max_length=50)
    schema.add_field("start_char",      DataType.INT64)
    schema.add_field("end_char",        DataType.INT64)
    schema.add_field("token_count",     DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="FLAT",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=MILVUS_COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    print(f"Collection '{MILVUS_COLLECTION}' created.")


def upsert_chunks(chunks: list[dict]):
    client = get_client()

    rows = []
    for c in chunks:
        rows.append({
            "chunk_id":        c["chunk_id"],
            "embedding":       c["embedding"],
            "doc_id":          c["doc_id"],
            "session_id":      c.get("session_id", ""),
            "filename":        c["filename"],
            "text":            c["text"][:4000],
            "section_heading": (c["section_heading"] or "")[:500],
            "page":            str(c["page"]) if c["page"] is not None else "",
            "start_char":      c["start_char"],
            "end_char":        c["end_char"],
            "token_count":     c["token_count"],
        })

    client.upsert(collection_name=MILVUS_COLLECTION, data=rows)
    print(f"Upserted {len(rows)} chunks.")


def search(
    query_vector: list[float],
    session_id:   str,
    top_k:        int = 5,
    doc_id:       str | None = None,
) -> list[dict]:
    """
    Search by session_id (across all docs in session).
    Optionally scope to a single doc_id for edit operations.
    """
    client = get_client()

    if doc_id:
        filter_expr = f'session_id == "{session_id}" && doc_id == "{doc_id}"'
    else:
        filter_expr = f'session_id == "{session_id}"'

    results = client.search(
        collection_name=MILVUS_COLLECTION,
        data=[query_vector],
        filter=filter_expr,
        limit=top_k,
        output_fields=[
            "chunk_id", "doc_id", "session_id", "filename", "text",
            "section_heading", "page", "start_char", "end_char",
        ],
    )

    hits = []
    for hit in results[0]:
        entity = hit["entity"]
        entity["score"] = hit["distance"]
        hits.append(entity)

    return hits


def get_all_chunks(session_id: str, doc_id: str | None = None) -> list[dict]:
    """
    Retrieve all chunks for a session (or single doc within session).
    Used for summarise intent.
    """
    client = get_client()

    if doc_id:
        filter_expr = f'session_id == "{session_id}" && doc_id == "{doc_id}"'
    else:
        filter_expr = f'session_id == "{session_id}"'

    return client.query(
        collection_name=MILVUS_COLLECTION,
        filter=filter_expr,
        output_fields=[
            "chunk_id", "doc_id", "session_id", "filename", "text",
            "section_heading", "page", "start_char", "end_char", "token_count",
        ],
    )


def delete_session(session_id: str):
    client = get_client()
    client.delete(
        collection_name=MILVUS_COLLECTION,
        filter=f'session_id == "{session_id}"',
    )
    print(f"Deleted all chunks for session_id='{session_id}'.")


def delete_document(doc_id: str):
    client = get_client()
    client.delete(
        collection_name=MILVUS_COLLECTION,
        filter=f'doc_id == "{doc_id}"',
    )
    print(f"Deleted all chunks for doc_id='{doc_id}'.")