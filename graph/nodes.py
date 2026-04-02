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
- summarise : user wants a summary of the whole document
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


# ── Retriever ─────────────────────────────────────────────────────────────────

def retriever_node(state: DocState) -> dict:
    query  = state["messages"][-1].content
    doc_id = state["doc_id"]
    intent = state["intent"]

    if intent == "summarise":
        chunks = get_all_chunks(doc_id)
    else:
        q_vec  = embed_query(query)
        chunks = search(q_vec, doc_id, top_k=5)

    print(f"[retriever] fetched {len(chunks)} chunks  (intent={intent})")
    return {"retrieved_chunks": chunks}


# ── Generate ──────────────────────────────────────────────────────────────────

def generate_node(state: DocState) -> dict:
    query  = state["messages"][-1].content
    intent = state["intent"]
    chunks = state["retrieved_chunks"]

    context = "\n\n---\n\n".join(
        f"[Section: {c.get('section_heading', '')}  Page: {c.get('page', '')}]\n{c['text']}"
        for c in chunks
    )

    prompts = {
        "summarise": (
            "You are a document assistant. Summarise the following document sections "
            "into a clear, concise summary. Use bullet points for key facts.",
            f"Document sections:\n\n{context}",
        ),
        "explain": (
            "You are a document assistant. Explain the following content in simple, "
            "easy-to-understand language as if explaining to a non-expert.",
            f"Content:\n\n{context}\n\nUser asked: {query}",
        ),
        "qa": (
            "You are a document assistant. Answer the user's question using only "
            "the context provided. If the answer is not in the context, say so.",
            f"Context:\n\n{context}\n\nQuestion: {query}",
        ),
    }

    system, user = prompts.get(intent, prompts["qa"])
    response = _chat(system, user)

    print(f"[generate] response length={len(response)} chars")
    return {"response": response}


# ── Edit ──────────────────────────────────────────────────────────────────────

def edit_node(state: DocState) -> dict:
    query        = state["messages"][-1].content
    doc_id       = state["doc_id"]
    filename     = state["filename"]
    current_text = state.get("current_text", "")
    edit_history = state.get("edit_history", [])

    # step 1 — find the most relevant chunk for this edit instruction
    q_vec = embed_query(query)
    hits  = search(q_vec, doc_id, top_k=1)

    if not hits:
        return {"response": "Could not find a relevant section to edit."}

    target = hits[0]
    original_chunk_text = target["text"]
    start_char          = target["start_char"]
    end_char            = target["end_char"]
    chunk_id            = target["chunk_id"]
    section_heading     = target.get("section_heading", "")
    page                = target.get("page", "")

    print(f"[edit] target chunk: {chunk_id}  section: {section_heading}")
    print(f"[edit] original text preview: {original_chunk_text[:80]}...")

    # step 2 — ask LLM to produce the edited version of just this chunk
    edited_chunk_text = _chat(
        system="""You are a document editor. 
The user will give you a piece of text and an edit instruction.
Return ONLY the edited text with the instruction applied.
Do not add explanations, commentary, or formatting.
Preserve the original structure and style unless instructed otherwise.""",
        user=f"""Original text:
{original_chunk_text}

Edit instruction:
{query}

Return only the edited text:""",
    )

    print(f"[edit] edited text preview: {edited_chunk_text[:80]}...")

    # step 3 — apply the diff to current_text using char offsets
    if not current_text:
        # first edit — current_text not yet set, will be patched on export
        current_text = ""

    # record the edit in history
    edit_record = {
        "chunk_id":      chunk_id,
        "doc_id":        doc_id,
        "start_char":    start_char,
        "end_char":      end_char,
        "original_text": original_chunk_text,
        "edited_text":   edited_chunk_text,
        "instruction":   query,
        "section":       section_heading,
    }
    edit_history = edit_history + [edit_record]

    # step 4 — re-embed the edited chunk and upsert back to Milvus
    new_embedding = embed_texts([edited_chunk_text])[0]

    updated_chunk = {
        "chunk_id":        chunk_id,
        "doc_id":          doc_id,
        "filename":        filename,
        "text":            edited_chunk_text[:4000],
        "section_heading": section_heading,
        "page":            page,
        "start_char":      start_char,
        "end_char":        end_char,
        "token_count":     len(edited_chunk_text.split()),
        "embedding":       new_embedding,
    }
    upsert_chunks([updated_chunk])
    print(f"[edit] re-embedded and upserted chunk {chunk_id}")

    response = (
        f"Done. I've edited the '{section_heading}' section.\n\n"
        f"**Original:**\n{original_chunk_text[:300]}...\n\n"
        f"**Edited:**\n{edited_chunk_text[:300]}..."
    )

    return {
        "response":     response,
        "edit_history": edit_history,
        "current_text": current_text,
    }