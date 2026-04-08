from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_LLM_KEY,
    AZURE_LLM_ENDPOINT,
    AZURE_LLM_API_VERSION,
    AZURE_LLM_DEPLOYMENT,
)
from vectorstore.embedder import embed_query, embed_texts
from vectorstore.milvus_client import search, get_all_chunks, upsert_chunks, search_per_doc
from api.session import get_docs
from graph.state import DocState

_llm = AzureOpenAI(
    api_key=AZURE_OPENAI_LLM_KEY,
    azure_endpoint=AZURE_LLM_ENDPOINT,
    api_version=AZURE_LLM_API_VERSION,
)

_MAX_SUMMARISE_CHUNKS = 50
_MAX_QA_CHUNKS        = 6
_MAX_CONTEXT_TOKENS   = 80_000


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
- general   : greetings, small talk, asking what the bot can do, off-topic questions
- summarise : user wants a summary of the whole document or documents
- explain   : user wants something explained in simple terms
- qa        : user is asking a factual question about the document
- edit      : user wants to modify, rephrase, delete or add text
- compare   : user wants to compare two or more documents against each other
- analyse   : user wants cross-document insights, patterns, contradictions, or trends

Reply with only the label, nothing else.""",
        user=last_msg,
    )

    intent = intent.strip().lower()
    if intent not in ("general", "summarise", "explain", "qa", "edit", "compare", "analyse"):
        intent = "qa"

    print(f"[intent] → {intent}")
    return {"intent": intent}

# ── General handler — no retrieval needed ────────────────────────────────────

def general_node(state: DocState) -> dict:
    last_msg   = state["messages"][-1].content
    doc_names  = state.get("filename", "your uploaded documents")

    response = _chat(
        system="""You are a helpful document assistant. 
Your job is to help users understand, summarise, explain, and edit their uploaded documents.
You can:
- Answer questions about uploaded documents
- Summarise documents
- Explain content in simple terms
- Edit or rephrase sections of documents

You cannot:
- Answer general knowledge questions unrelated to uploaded documents
- Browse the internet
- Remember conversations from previous sessions

Keep your response short, friendly and focused on what you can actually do.
Do not mention any specific document content in your greeting.""",
        user=last_msg,
    )

    return {
        "response": response,
        "sources":  [],
    }


# ── Retriever ─────────────────────────────────────────────────────────────────

def retriever_node(state: DocState) -> dict:
    query      = state["messages"][-1].content
    session_id = state["session_id"]
    intent     = state["intent"]

    if intent == "summarise":
        all_chunks = get_all_chunks(session_id=session_id)
        if len(all_chunks) > _MAX_SUMMARISE_CHUNKS:
            step   = len(all_chunks) // _MAX_SUMMARISE_CHUNKS
            chunks = all_chunks[::step][:_MAX_SUMMARISE_CHUNKS]
        else:
            chunks = all_chunks
        print(f"[retriever] summarise: {len(all_chunks)} total → {len(chunks)} sampled")
    else:
        q_vec  = embed_query(query)
        chunks = search(q_vec, session_id=session_id, top_k=_MAX_QA_CHUNKS)

    print(f"[retriever] fetched {len(chunks)} chunks  (intent={intent})")
    return {"retrieved_chunks": chunks}


# ── Context builder ───────────────────────────────────────────────────────────

def _trim_context(chunks: list[dict], max_tokens: int = _MAX_CONTEXT_TOKENS) -> str:
    max_chars = max_tokens * 4
    parts     = []
    total     = 0

    for c in chunks:
        piece = (
            f"[Source: {c.get('filename', 'unknown')}  "
            f"Section: {c.get('section_heading', '')}  "
            f"Page: {c.get('page', '')}]\n{c['text']}"
        )
        if total + len(piece) > max_chars:
            print(f"[generate] context trimmed at {len(parts)} chunks")
            break
        parts.append(piece)
        total += len(piece)

    return "\n\n---\n\n".join(parts)


# ── Generate ──────────────────────────────────────────────────────────────────

def generate_node(state: DocState) -> dict:
    query  = state["messages"][-1].content
    intent = state["intent"]
    chunks = state["retrieved_chunks"]
    memory = state.get("memory", [])

    context = _trim_context(chunks)
    sources = list(dict.fromkeys(c.get("filename", "") for c in chunks))

    memory_str = ""
    if memory:
        memory_str = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in memory
        )
        memory_str = f"Conversation so far:\n{memory_str}\n\n"

    # guardrail system prompt applied to all intents
    guardrail = """You are a strict document assistant. 
RULES you must follow at all times:
- Only answer based on the provided document context
- If the answer is not in the context, say clearly: "I could not find this information in the uploaded documents."
- Do not use your general knowledge to fill in gaps
- Do not make up or infer information not present in the context
- Do not answer questions unrelated to the uploaded documents
- Always mention which document the answer came from"""

    prompts = {
        "summarise": (
            guardrail + "\nSummarise the provided document sections clearly and concisely "
            "using bullet points. Group by document name if multiple documents.",
            f"{memory_str}Document sections:\n\n{context}",
        ),
        "explain": (
            guardrail + "\nExplain the following content in simple, easy-to-understand "
            "language. Mention which document the information came from.",
            f"{memory_str}Content:\n\n{context}\n\nUser asked: {query}",
        ),
        "qa": (
            guardrail + "\nAnswer the user's question using ONLY the context below. "
            "If multiple documents contribute, mention each by name.",
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

    q_vec = embed_query(query)
    hits  = search(q_vec, session_id=session_id, top_k=3)

    if not hits:
        return {
            "response": "Could not find a relevant section to edit.",
            "sources":  [],
        }

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
            "response": "I could not find that specific content to edit. "
                        "Please be more specific.",
            "sources":  [filename],
        }

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
    
    
# ── Compare node ──────────────────────────────────────────────────────────────

def compare_node(state: DocState) -> dict:
    """
    Fetch relevant chunks from each document separately
    and ask LLM to compare them side by side.
    """
    

    query      = state["messages"][-1].content
    session_id = state["session_id"]
    memory     = state.get("memory", [])

    # get all docs in session
    docs = get_docs(session_id)

    if len(docs) < 2:
        return {
            "response": "Please upload at least two documents to compare.",
            "sources":  [],
        }

    # embed query once
    q_vec = embed_query(query)

    # search each doc separately — 4 chunks per doc
    per_doc_results = search_per_doc(
        query_vector=q_vec,
        session_id=session_id,
        doc_ids=[d["doc_id"] for d in docs],
        top_k=4,
    )

    # build side-by-side context block
    doc_sections = []
    sources      = []

    for d in docs:
        doc_id   = d["doc_id"]
        filename = d["filename"]
        chunks   = per_doc_results.get(doc_id, [])
        sources.append(filename)

        if not chunks:
            doc_sections.append(
                f"=== {filename} ===\nNo relevant content found."
            )
            continue

        content = "\n\n".join(
            f"[Section: {c.get('section_heading','')}  Page: {c.get('page','')}]\n{c['text']}"
            for c in chunks
        )
        doc_sections.append(f"=== {filename} ===\n{content}")

    combined_context = "\n\n\n".join(doc_sections)

    memory_str = ""
    if memory:
        memory_str = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in memory
        )
        memory_str = f"Conversation so far:\n{memory_str}\n\n"

    response = _chat(
        system="""You are a document comparison specialist.
You will be given content from multiple documents clearly separated by document name.
Your job is to:
- Compare and contrast the documents on the topic asked
- Highlight key similarities
- Highlight key differences
- Note anything that contradicts between documents
- Be specific — always reference the document name when making a point
- Only use information present in the provided content
- If a document does not contain relevant information on the topic, say so explicitly""",
        user=(
            f"{memory_str}"
            f"Compare the following documents on this topic: {query}\n\n"
            f"{combined_context}"
        ),
    )

    print(f"[compare] docs={[d['filename'] for d in docs]}  sources={sources}")
    return {
        "response": response,
        "sources":  sources,
    }


# ── Analyse node ──────────────────────────────────────────────────────────────

def analyse_node(state: DocState) -> dict:
    """
    Cross-document analysis — patterns, contradictions, insights across all docs.
    Fetches a sample from every document and asks LLM to find patterns.
    """
    

    query      = state["messages"][-1].content
    session_id = state["session_id"]
    memory     = state.get("memory", [])

    docs = get_docs(session_id)

    if len(docs) < 2:
        return {
            "response": "Please upload at least two documents to analyse across.",
            "sources":  [],
        }

    q_vec           = embed_query(query)
    per_doc_results = search_per_doc(
        query_vector=q_vec,
        session_id=session_id,
        doc_ids=[d["doc_id"] for d in docs],
        top_k=5,
    )

    doc_sections = []
    sources      = []

    for d in docs:
        doc_id   = d["doc_id"]
        filename = d["filename"]
        chunks   = per_doc_results.get(doc_id, [])
        sources.append(filename)

        if not chunks:
            doc_sections.append(
                f"=== {filename} ===\nNo relevant content found."
            )
            continue

        content = "\n\n".join(
            f"[Section: {c.get('section_heading','')}]\n{c['text']}"
            for c in chunks
        )
        doc_sections.append(f"=== {filename} ===\n{content}")

    combined_context = "\n\n\n".join(doc_sections)

    memory_str = ""
    if memory:
        memory_str = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in memory
        )
        memory_str = f"Conversation so far:\n{memory_str}\n\n"

    response = _chat(
        system="""You are a cross-document analysis specialist.
You will be given content from multiple documents.
Your job is to:
- Identify common themes and patterns across documents
- Find contradictions or inconsistencies between documents
- Extract insights that only become visible when looking across all documents
- Identify gaps — things mentioned in one document but missing in others
- Always reference the specific document name for every point you make
- Only use information present in the provided content
- Structure your response clearly with sections: Patterns, Contradictions, Insights, Gaps""",
        user=(
            f"{memory_str}"
            f"Analyse across all documents: {query}\n\n"
            f"{combined_context}"
        ),
    )

    print(f"[analyse] docs={[d['filename'] for d in docs]}  sources={sources}")
    return {
        "response": response,
        "sources":  sources,
    }