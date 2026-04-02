from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_EMB_KEY,
    AZURE_EMB_ENDPOINT,
    AZURE_EMB_API_VERSION,
    AZURE_EMB_DEPLOYMENT,
)

_client = AzureOpenAI(
    api_key=AZURE_OPENAI_EMB_KEY,
    azure_endpoint=AZURE_EMB_ENDPOINT,
    api_version=AZURE_EMB_API_VERSION,
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings in one API call.
    Returns a list of vectors in the same order as input.
    """
    response = _client.embeddings.create(
        model=AZURE_EMB_DEPLOYMENT,
        input=texts,
    )
    # sort by index to guarantee order
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Attach an 'embedding' field to each chunk dict in place.
    Returns the same list with embeddings added.
    """
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string for retrieval.
    """
    return embed_texts([query])[0]