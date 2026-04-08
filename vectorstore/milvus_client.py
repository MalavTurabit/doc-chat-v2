from pymilvus import MilvusClient, DataType
from config import MILVUS_DB_PATH, MILVUS_COLLECTION, EMBEDDING_DIM
import logging

logger = logging.getLogger(__name__)

_client: MilvusClient | None = None

# upsert in batches to avoid gRPC too_many_pings
_UPSERT_BATCH_SIZE = 500


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(MILVUS_DB_PATH)
    return _client


def init_collection():
    client = get_client()

    if client.has_collection(MILVUS_COLLECTION):
        logger.info(f"Collection '{MILVUS_COLLECTION}' already exists.")
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
    logger.info(f"Collection '{MILVUS_COLLECTION}' created.")


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
            "section_heading": (c.get("section_heading") or "")[:500],
            "page":            str(c["page"]) if c["page"] is not None else "",
            "start_char":      c["start_char"],
            "end_char":        c["end_char"],
            "token_count":     c["token_count"],
        })

    # batch upsert to avoid gRPC connection spam
    total_batches = (len(rows) + _UPSERT_BATCH_SIZE - 1) // _UPSERT_BATCH_SIZE
    for i in range(0, len(rows), _UPSERT_BATCH_SIZE):
        batch = rows[i: i + _UPSERT_BATCH_SIZE]
        client.upsert(collection_name=MILVUS_COLLECTION, data=batch)
        logger.info(
            f"[milvus] upsert batch "
            f"{i // _UPSERT_BATCH_SIZE + 1}/{total_batches} "
            f"— {len(batch)} chunks"
        )


def search(
    query_vector: list[float],
    session_id:   str,
    top_k:        int = 5,
    doc_id:       str | None = None,
) -> list[dict]:
    client = get_client()

    filter_expr = (
        f'session_id == "{session_id}" && doc_id == "{doc_id}"'
        if doc_id
        else f'session_id == "{session_id}"'
    )

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
        entity        = hit["entity"]
        entity["score"] = hit["distance"]
        hits.append(entity)

    return hits


def get_all_chunks(session_id: str, doc_id: str | None = None) -> list[dict]:
    client = get_client()

    filter_expr = (
        f'session_id == "{session_id}" && doc_id == "{doc_id}"'
        if doc_id
        else f'session_id == "{session_id}"'
    )

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
    logger.info(f"[milvus] deleted session '{session_id}'")


def delete_document(doc_id: str):
    client = get_client()
    client.delete(
        collection_name=MILVUS_COLLECTION,
        filter=f'doc_id == "{doc_id}"',
    )
    logger.info(f"[milvus] deleted doc '{doc_id}'")
    

def search_per_doc(
    query_vector: list[float],
    session_id:   str,
    doc_ids:      list[str],
    top_k:        int = 3,
) -> dict[str, list[dict]]:
    """
    Search each document separately and return results keyed by doc_id.
    Used for side-by-side comparison.
    """
    client  = get_client()
    results = {}

    for doc_id in doc_ids:
        hits = client.search(
            collection_name=MILVUS_COLLECTION,
            data=[query_vector],
            filter=f'session_id == "{session_id}" && doc_id == "{doc_id}"',
            limit=top_k,
            output_fields=[
                "chunk_id", "doc_id", "session_id", "filename", "text",
                "section_heading", "page", "start_char", "end_char",
            ],
        )
        doc_hits = []
        for hit in hits[0]:
            entity          = hit["entity"]
            entity["score"] = hit["distance"]
            doc_hits.append(entity)
        results[doc_id] = doc_hits

    return results