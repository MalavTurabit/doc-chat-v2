from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_LLM_KEY,
    AZURE_LLM_ENDPOINT,
    AZURE_LLM_API_VERSION,
    AZURE_LLM_DEPLOYMENT,
)
from vectorstore.embedder import embed_query, embed_texts
from vectorstore.milvus_client import search, get_all_chunks, upsert_chunks
from graph.state import DocState

_llm = AzureOpenAI(
    api_key=AZURE_OPENAI_LLM_KEY,
    azure_endpoint=AZURE_LLM_ENDPOINT,
    api_version=AZURE_LLM_API_VERSION,
)


def _chat(system: str, user: str) -> str:
    response = _llm.chat.completions.create(
        model=AZURE_LLM_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


# ── Intent classifier ─────────────────────────────────────────────────────────

def classify_intent(state: DocState) -> dict:
    last_msg = state["messages"][-1].content

    intent = _chat(
        system="""Classify the user message into exactly one of these labels:
- summarise : user wants a summary of the whole document or documents
- explain   : user wants something explained in simple terms
- qa        : user is asking a factual question about the document
- edit      : user wants to modify, rephrase, delete or add text

Reply with only the label, nothing else.""",
        user=last_msg,
    )

    intent = intent.strip().lower()
    if intent not in ("summarise", "explain", "qa", "edit"):
        intent = "qa"

    print(f"[intent] → {intent}")
    return {"intent": intent}


# ── Retriever — searches ALL docs in session ──────────────────────────────────

def retriever_node(state: DocState) -> dict:
    query      = state["messages"][-1].content
    session_id = state["session_id"]
    intent     = state["intent"]

    if intent == "summarise":
        # get all chunks across all docs in session
        chunks = get_all_chunks(session_id=session_id)
    else:
        q_vec  = embed_query(query)
        chunks = search(q_vec, session_id=session_id, top_k=6)

    print(f"[retriever] fetched {len(chunks)} chunks  (intent={intent})")
    return {"retrieved_chunks": chunks}


# ── Generate — cites source filename in answer ────────────────────────────────

def generate_node(state: DocState) -> dict:
    query      = state["messages"][-1].content
    intent     = state["intent"]
    chunks     = state["retrieved_chunks"]
    memory     = state.get("memory", [])

    # build context with source attribution per chunk
    context = "\n\n---\n\n".join(
        f"[Source: {c.get('filename', 'unknown')}  "
        f"Section: {c.get('section_heading', '')}  "
        f"Page: {c.get('page', '')}]\n{c['text']}"
        for c in chunks
    )

    # collect unique source filenames for metadata
    sources = list(dict.fromkeys(c.get("filename", "") for c in chunks))

    # build memory context string
    memory_str = ""
    if memory:
        memory_str = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in memory
        )
        memory_str = f"Conversation so far:\n{memory_str}\n\n"

    prompts = {
        "summarise": (
            "You are a document assistant. Summarise the provided document sections "
            "clearly and concisely using bullet points. "
            "When content comes from multiple documents, group by document and "
            "mention the document name as a heading.",
            f"{memory_str}Document sections:\n\n{context}",
        ),
        "explain": (
            "You are a document assistant. Explain the following content in simple "
            "easy-to-understand language. "
            "At the end of your answer mention which document(s) the information came from.",
            f"{memory_str}Content:\n\n{context}\n\nUser asked: {query}",
        ),
        "qa": (
            "You are a document assistant. Answer the user's question using only "
            "the context provided. "
            "If the answer comes from multiple documents, mention each document by name. "
            "If the answer is not in the context, say so clearly.",
            f"{memory_str}Context:\n\n{context}\n\nQuestion: {query}",
        ),
    }

    system, user = prompts.get(intent, prompts["qa"])
    response = _chat(system, user)

    print(f"[generate] response length={len(response)} chars  sources={sources}")
    return {
        "response": response,
        "sources":  sources,
    }


# ── Edit ──────────────────────────────────────────────────────────────────────

def edit_node(state: DocState) -> dict:
    query      = state["messages"][-1].content
    session_id = state["session_id"]
    memory     = state.get("memory", [])

    # step 1 — retrieve top 3 chunks across session
    q_vec = embed_query(query)
    hits  = search(q_vec, session_id=session_id, top_k=3)

    if not hits:
        return {
            "response":  "Could not find a relevant section to edit.",
            "sources":   [],
        }

    # step 2 — let LLM pick the most relevant chunk
    chunks_preview = "\n\n".join(
        f"[{i}] file='{h.get('filename', '')}' "
        f"section='{h.get('section_heading', '')}'\n{h['text'][:300]}"
        for i, h in enumerate(hits)
    )

    pick_response = _chat(
        system="""You are a document editor assistant.
Given an edit instruction and several document chunks, reply with ONLY
the index number (0, 1, or 2) of the chunk that should be edited.
Reply with a single digit only.""",
        user=f"Edit instruction: {query}\n\nChunks:\n{chunks_preview}\n\n"
             f"Which chunk index (0, 1, or 2) should be edited?",
    )

    try:
        pick = int(pick_response.strip()[0])
        if pick not in range(len(hits)):
            pick = 0
    except Exception:
        pick = 0

    target              = hits[pick]
    original_chunk_text = target["text"]
    start_char          = target["start_char"]
    end_char            = target["end_char"]
    chunk_id            = target["chunk_id"]
    section_heading     = target.get("section_heading", "")
    page                = target.get("page", "")
    filename            = target.get("filename", "")
    doc_id              = target.get("doc_id", "")

    print(f"[edit] picked chunk {pick}  file={filename}  section={section_heading}")
    print(f"[edit] FULL original:\n{original_chunk_text}\n")

    # step 3 — apply the edit strictly
    edited_chunk_text = _chat(
        system="""You are a document editor.
Rules:
- Apply ONLY the specific change requested
- Do NOT rewrite, reformat, or add new content
- Do NOT invent information not present in the original
- Preserve all other text exactly as-is
- Return ONLY the edited text, nothing else""",
        user=f"Original text:\n{original_chunk_text}\n\n"
             f"Edit instruction:\n{query}\n\n"
             f"Return only the edited text with the single change applied:",
    )

    print(f"[edit] FULL edited:\n{edited_chunk_text}\n")

    if edited_chunk_text.strip() == original_chunk_text.strip():
        return {
            "response":  "I could not find that specific content to edit. "
                         "Please be more specific.",
            "sources":   [filename],
        }

    # step 4 — record edit
    edit_record = {
        "chunk_id":      chunk_id,
        "doc_id":        doc_id,
        "session_id":    session_id,
        "filename":      filename,
        "start_char":    start_char,
        "end_char":      end_char,
        "original_text": original_chunk_text,
        "edited_text":   edited_chunk_text,
        "instruction":   query,
        "section":       section_heading,
    }

    # step 5 — re-embed and upsert
    new_embedding = embed_texts([edited_chunk_text])[0]
    upsert_chunks([{
        "chunk_id":        chunk_id,
        "doc_id":          doc_id,
        "session_id":      session_id,
        "filename":        filename,
        "text":            edited_chunk_text[:4000],
        "section_heading": section_heading,
        "page":            page,
        "start_char":      start_char,
        "end_char":        end_char,
        "token_count":     len(edited_chunk_text.split()),
        "embedding":       new_embedding,
    }])

    response = (
        f"Done. Edited **'{section_heading}'** in `{filename}`.\n\n"
        f"**Original:**\n{original_chunk_text[:300]}...\n\n"
        f"**Edited:**\n{edited_chunk_text[:300]}..."
    )

    return {
        "response":    response,
        "sources":     [filename],
        "edit_record": edit_record,
    }