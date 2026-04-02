import tiktoken
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def chunk_document(doc: dict) -> list[dict]:
    """
    Paragraph-aware greedy chunker.
    Receives the doc dict from parser.extract() and returns a list of chunks.
    Each chunk carries the text, metadata, and char offsets from the source doc.
    """
    blocks     = doc["blocks"]
    doc_id     = doc["doc_id"]
    filename   = doc["filename"]
    chunks     = []

    current_blocks   = []
    current_tokens   = 0
    current_heading  = None

    def flush(current_blocks, current_heading):
        if not current_blocks:
            return
        text       = "\n".join(b["text"] for b in current_blocks)
        start_char = current_blocks[0]["start_char"]
        end_char   = current_blocks[-1]["end_char"]
        page       = current_blocks[0]["page"]

        chunks.append({
            "chunk_id":        None,           # assigned in milvus_client
            "doc_id":          doc_id,
            "filename":        filename,
            "text":            text,
            "section_heading": current_heading,
            "page":            page,
            "start_char":      start_char,
            "end_char":        end_char,
            "token_count":     count_tokens(text),
        })

    for block in blocks:

        # headings update the running section label but are not chunked alone
        if block["type"] == "heading":
            current_heading = block["text"]
            continue

        block_tokens = count_tokens(block["text"])

        # tables always get their own chunk regardless of size
        if block["type"] == "table":
            flush(current_blocks, current_heading)
            current_blocks  = []
            current_tokens  = 0
            chunks.append({
                "chunk_id":        None,
                "doc_id":          doc_id,
                "filename":        filename,
                "text":            block["text"],
                "section_heading": current_heading,
                "page":            block["page"],
                "start_char":      block["start_char"],
                "end_char":        block["end_char"],
                "token_count":     block_tokens,
            })
            continue

        # if adding this block exceeds the limit, flush and start fresh
        if current_tokens + block_tokens > CHUNK_SIZE_TOKENS:
            flush(current_blocks, current_heading)

            # overlap: carry the last block into the new chunk
            if current_blocks:
                overlap_block  = current_blocks[-1]
                current_blocks = [overlap_block]
                current_tokens = count_tokens(overlap_block["text"])
            else:
                current_blocks = []
                current_tokens = 0

        current_blocks.append(block)
        current_tokens += block_tokens

    # flush whatever is left
    flush(current_blocks, current_heading)

    # assign sequential chunk IDs
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"{doc_id}_{i}"

    return chunks