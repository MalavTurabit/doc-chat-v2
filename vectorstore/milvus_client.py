from pymilvus import MilvusClient, DataType
from config import MILVUS_DB_PATH, MILVUS_COLLECTION, EMBEDDING_DIM

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(MILVUS_DB_PATH)
    return _client


def init_collection():
    """
    Create the collection with explicit schema if it does not exist.
    """
    client = get_client()

    if client.has_collection(MILVUS_COLLECTION):
        print(f"Collection '{MILVUS_COLLECTION}' already exists — skipping creation.")
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

    schema.add_field("chunk_id",        DataType.VARCHAR, max_length=200, is_primary=True)
    schema.add_field("embedding",       DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("doc_id",          DataType.VARCHAR, max_length=100)
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
            "filename":        c["filename"],
            "text":            c["text"][:4000],
            "section_heading": (c["section_heading"] or "")[:500],
            "page":            str(c["page"]) if c["page"] is not None else "",
            "start_char":      c["start_char"],
            "end_char":        c["end_char"],
            "token_count":     c["token_count"],
        })

    client.upsert(collection_name=MILVUS_COLLECTION, data=rows)
    print(f"Upserted {len(rows)} chunks into '{MILVUS_COLLECTION}'.")


def search(query_vector: list[float], doc_id: str, top_k: int = 5) -> list[dict]:
    client = get_client()

    results = client.search(
        collection_name=MILVUS_COLLECTION,
        data=[query_vector],
        filter=f'doc_id == "{doc_id}"',
        limit=top_k,
        output_fields=[
            "chunk_id", "doc_id", "filename", "text",
            "section_heading", "page", "start_char", "end_char",
        ],
    )

    hits = []
    for hit in results[0]:
        entity = hit["entity"]
        entity["score"] = hit["distance"]
        hits.append(entity)

    return hits


def delete_document(doc_id: str):
    client = get_client()
    client.delete(
        collection_name=MILVUS_COLLECTION,
        filter=f'doc_id == "{doc_id}"',
    )
    print(f"Deleted all chunks for doc_id='{doc_id}'.")


def get_all_chunks(doc_id: str) -> list[dict]:
    client = get_client()
    results = client.query(
        collection_name=MILVUS_COLLECTION,
        filter=f'doc_id == "{doc_id}"',
        output_fields=[
            "chunk_id", "doc_id", "filename", "text",
            "section_heading", "page", "start_char", "end_char", "token_count",
        ],
    )
    return results