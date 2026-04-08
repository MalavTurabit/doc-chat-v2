from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_EMB_KEY,
    AZURE_EMB_ENDPOINT,
    AZURE_EMB_API_VERSION,
    AZURE_EMB_DEPLOYMENT,
)
import logging
import time
import tiktoken

logger = logging.getLogger(__name__)

_client = AzureOpenAI(
    api_key=AZURE_OPENAI_EMB_KEY,
    azure_endpoint=AZURE_EMB_ENDPOINT,
    api_version=AZURE_EMB_API_VERSION,
)

_enc            = tiktoken.get_encoding("cl100k_base")
_BATCH_SIZE     = 50
_BATCH_DELAY    = 0.4
_MAX_TOKENS     = 8000   # stay safely under the 8192 limit


def _truncate(text: str) -> str:
    """Truncate text to fit within the model token limit."""
    tokens = _enc.encode(text)
    if len(tokens) <= _MAX_TOKENS:
        return text
    truncated = _enc.decode(tokens[:_MAX_TOKENS])
    logger.warning(
        f"[embed] chunk truncated from {len(tokens)} to {_MAX_TOKENS} tokens"
    )
    return truncated


def embed_texts(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    total_batches  = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE

    # truncate any oversized texts before sending
    texts = [_truncate(t) for t in texts]

    for i in range(0, len(texts), _BATCH_SIZE):
        batch     = texts[i: i + _BATCH_SIZE]
        batch_num = i // _BATCH_SIZE + 1
        logger.info(
            f"[embed] batch {batch_num}/{total_batches} — {len(batch)} texts"
        )

        response = _client.embeddings.create(
            model=AZURE_EMB_DEPLOYMENT,
            input=batch,
        )
        batch_embeddings = [
            item.embedding
            for item in sorted(response.data, key=lambda x: x.index)
        ]
        all_embeddings.extend(batch_embeddings)

        if batch_num < total_batches:
            time.sleep(_BATCH_DELAY)

    return all_embeddings


def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts   = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]