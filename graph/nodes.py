from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_LLM_KEY,
    AZURE_LLM_ENDPOINT,
    AZURE_LLM_API_VERSION,
    AZURE_LLM_DEPLOYMENT,
)
from vectorstore.embedder import embed_query, embed_texts
from vectorstore.milvus_client import search, get_all_chunks, upsert_chunks, search_per_doc ,keyword_search , search_by_page, search_by_section
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
- general   : pure greetings (hi, hello), asking what the bot can do, completely off-topic questions unrelated to any document
- summarise : user wants a summary or overview of the document(s) — includes "what is this doc", "what's in the file", "give me an overview"
- explain   : user wants something specific explained in simple terms
- qa        : user is asking a specific factual question about document content
- edit      : user wants to modify, rephrase, delete or add text
- compare   : user wants to compare two or more documents against each other
- analyse   : user wants cross-document insights, patterns, contradictions, or trends

Important rules:
- "what is this doc", "whats the doc", "what does this file contain" → summarise
- "hi", "hello", "what can you do" → general
- Any question about document content → qa
- When in doubt between general and qa/summarise → choose qa or summarise

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
    session_id = state["session_id"]

    docs      = get_docs(session_id)
    doc_names = [d["filename"] for d in docs]

    if doc_names:
        doc_context = f"The user has uploaded these documents: {', '.join(doc_names)}."
    else:
        doc_context = "No documents have been uploaded yet."

    response = _chat(
        system=f"""You are a helpful document assistant.
{doc_context}

You can help the user with:
- Answering questions about their uploaded documents
- Summarising documents
- Explaining content in simple terms
- Editing or rephrasing sections
- Comparing documents against each other
- Cross-document analysis

You cannot answer general knowledge questions unrelated to the uploaded documents.
Keep your response short and friendly.
If documents are uploaded, encourage the user to ask questions about them.""",
        user=last_msg,
    )

    return {
        "response": response,
        "sources":  [],
    }

# -─ Regex indentifier ─────────────────────────────────────────────────────────────
import re

# ── Query rewriter (follow-up resolver) ───────────────────────────────────────

def _rewrite_query_if_followup(query: str, memory: list[dict]) -> str:
    """
    Detects follow-up queries and rewrites them using conversation memory.
    Only calls LLM if follow-up indicators are detected — saves cost.
    """
    if not memory:
        return query

    # fast regex check — follow-up indicators
    followup_patterns = [
        r'\b(he|she|they|it|his|her|their|its)\b',   # pronouns
        r'\b(that|this|those|these)\b',                # demonstratives
        r'\bsame\b',                                   # "same person"
        r'^(what about|how about|and|also|tell me more|what else)',  # continuations
        r'\b(previous|last|above|mentioned)\b',        # references
    ]

    is_followup = any(
        re.search(pattern, query.lower())
        for pattern in followup_patterns
    )

    if not is_followup:
        return query   # not a follow-up — return as-is, no LLM call

    # build memory context for rewriter
    last_turns = memory[-4:]   # last 2 turns is enough context
    memory_str = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}"
        for m in last_turns
    )

    rewritten = _chat(
        system="""You are a query rewriter.
Given a conversation history and a follow-up question, rewrite the follow-up
into a complete standalone question that can be understood without the conversation history.

Rules:
- Replace all pronouns with the actual entity they refer to
- Keep the question concise
- If it is already standalone, return it unchanged
- Return ONLY the rewritten question, nothing else""",
        user=f"""Conversation history:
{memory_str}

Follow-up question: {query}

Rewrite as standalone question:""",
    )

    print(f"[rewriter] '{query}' → '{rewritten}'")
    return rewritten


# ── Rule-based query type classifier ─────────────────────────────────────────

def _classify_query_type(query: str) -> str:
    """
    Classifies query type using regex rules. Zero LLM cost.
    Returns: identifier | name | analytical | positional | semantic
    """
    q = query.lower().strip()

    # ── identifier patterns ───────────────────────────────────────────────────
    # ISBN, numeric IDs, codes like EMP-001, TXN-9981, alphanumeric codes
    identifier_patterns = [
        r'\b\d{5,}\b',                      # 5+ digit number (ISBN, ID)
        r'\b[A-Z]{2,}-\d+\b',               # EMP-001, TXN-9981
        r'\b\d[\dX]{8,}\b',                 # ISBN-10/13 pattern
        r'\bisbn\b',                         # explicit ISBN mention
        r'\bcode\s*:?\s*\w+\b',             # "code: XYZ"
        r'\bid\s*:?\s*[\w\d]+\b',           # "id: 123"
    ]
    if any(re.search(p, query, re.IGNORECASE) for p in identifier_patterns):
        return "identifier"

    # ── analytical patterns ───────────────────────────────────────────────────
    analytical_keywords = [
        "average", "avg", "mean", "total", "sum", "count",
        "how many", "how much", "maximum", "minimum", "max", "min",
        "top 5", "top 10", "highest", "lowest", "most", "least",
        "percentage", "percent", "%", "ratio", "distribution",
        "trend", "growth", "compare numbers", "statistics", "stats",
        "aggregate", "group by", "breakdown",
    ]
    if any(kw in q for kw in analytical_keywords):
        return "analytical"

    # ── positional patterns ───────────────────────────────────────────────────
    positional_patterns = [
        r'\bpage\s+\d+\b',                  # page 5
        r'\bsection\s+[\d\.]+\b',           # section 3.2
        r'\bchapter\s+\d+\b',              # chapter 4
        r'\brow\s+\d+\b',                  # row 145
        r'\bslide\s+\d+\b',                # slide 3
        r'\bsheet\s+\w+\b',               # sheet Sales
        r'\bparagraph\s+\d+\b',           # paragraph 2
    ]
    if any(re.search(p, q) for p in positional_patterns):
        return "positional"

    # ── name patterns ─────────────────────────────────────────────────────────
    # Two or more capitalised words = likely a person/entity name
    name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
    if re.search(name_pattern, query):
        return "name"

    # ── default ───────────────────────────────────────────────────────────────
    return "semantic"


# ── Retriever ─────────────────────────────────────────────────────────────────

def retriever_node(state: DocState) -> dict:
    query      = state["messages"][-1].content
    session_id = state["session_id"]
    intent     = state["intent"]
    memory     = state.get("memory", [])

    # ── summarise — sample all chunks ────────────────────────────────────────
    if intent == "summarise":
        all_chunks = get_all_chunks(session_id=session_id)
        if len(all_chunks) > _MAX_SUMMARISE_CHUNKS:
            step   = len(all_chunks) // _MAX_SUMMARISE_CHUNKS
            chunks = all_chunks[::step][:_MAX_SUMMARISE_CHUNKS]
        else:
            chunks = all_chunks
        print(f"[retriever] summarise: {len(all_chunks)} total → {len(chunks)} sampled")
        return {"retrieved_chunks": chunks}

    # ── step 1: rewrite follow-up queries ────────────────────────────────────
    resolved_query = _rewrite_query_if_followup(query, memory)

    # ── step 2: classify query type ──────────────────────────────────────────
    query_type = _classify_query_type(resolved_query)
    print(f"[retriever] query_type={query_type}  resolved='{resolved_query}'")

    # ── step 3: retrieve based on type ───────────────────────────────────────

    if query_type == "semantic":
        chunks = search(
            embed_query(resolved_query),
            session_id=session_id,
            top_k=_MAX_QA_CHUNKS,
        )

    elif query_type in ("identifier", "name"):
        # semantic search first
        sem_chunks = search(
            embed_query(resolved_query),
            session_id=session_id,
            top_k=8,
        )

        # extract keywords based on type
        if query_type == "identifier":
            keywords = re.findall(r'\b\d[\dXx]{4,}\b', resolved_query)
            keywords += re.findall(r'\b[A-Z]{2,}-\d+\b', resolved_query)
        else:  # name
            keywords = re.findall(r'\b[A-Z][a-z]+\b', resolved_query)

        # keyword search for each keyword
        kw_chunks = []
        seen_ids  = {c["chunk_id"] for c in sem_chunks}
        for kw in keywords[:5]:
            hits = keyword_search(session_id, kw, top_k=5)
            for h in hits:
                if h["chunk_id"] not in seen_ids:
                    kw_chunks.append(h)
                    seen_ids.add(h["chunk_id"])

        chunks = sem_chunks + kw_chunks
        print(
            f"[retriever] hybrid: {len(sem_chunks)} semantic "
            f"+ {len(kw_chunks)} keyword  keywords={keywords[:5]}"
        )

    elif query_type == "analytical":
        # fetch all chunks — LLM needs full data to compute
        all_chunks = get_all_chunks(session_id=session_id)
        if len(all_chunks) > _MAX_SUMMARISE_CHUNKS:
            step   = len(all_chunks) // _MAX_SUMMARISE_CHUNKS
            chunks = all_chunks[::step][:_MAX_SUMMARISE_CHUNKS]
        else:
            chunks = all_chunks
        print(f"[retriever] analytical: {len(chunks)} chunks")

    elif query_type == "positional":
        # try to extract page/section number and filter by metadata
        page_match    = re.search(r'page\s+(\d+)', resolved_query, re.IGNORECASE)
        section_match = re.search(r'section\s+([\d\.]+)', resolved_query, re.IGNORECASE)

        if page_match:
            page_num = page_match.group(1)
            chunks   = search_by_page(session_id, page_num)
            print(f"[retriever] positional: page={page_num}  hits={len(chunks)}")
        elif section_match:
            section = section_match.group(1)
            chunks  = search_by_section(session_id, section)
            print(f"[retriever] positional: section={section}  hits={len(chunks)}")
        else:
            # fallback to semantic if we can't parse position
            chunks = search(
                embed_query(resolved_query),
                session_id=session_id,
                top_k=_MAX_QA_CHUNKS,
            )
    else:
        chunks = search(
            embed_query(resolved_query),
            session_id=session_id,
            top_k=_MAX_QA_CHUNKS,
        )

    print(f"[retriever] fetched {len(chunks)} chunks  (intent={intent}  type={query_type})")
    return {"retrieved_chunks": chunks, "query_type": query_type}

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
    query      = state["messages"][-1].content
    intent     = state["intent"]
    query_type = state.get("query_type", "semantic")
    chunks     = state["retrieved_chunks"]
    memory     = state.get("memory", [])

    context = _trim_context(chunks)
    sources = list(dict.fromkeys(c.get("filename", "") for c in chunks))

    memory_str = ""
    if memory:
        memory_str = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in memory
        )
        memory_str = f"Conversation so far:\n{memory_str}\n\n"

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
        "analytical": (
            guardrail + "\nThe user is asking an analytical question that may require "
            "computation or aggregation. Use the provided data to compute the answer. "
            "Show your working — list the values you used. "
            "Note: you are seeing a sample of the data, not necessarily all of it, "
            "so qualify your answer accordingly.",
            f"{memory_str}Data:\n\n{context}\n\nAnalytical question: {query}",
        ),
    }

    # use query_type for analytical, intent for everything else
    prompt_key   = "analytical" if query_type == "analytical" else intent
    system, user = prompts.get(prompt_key, prompts["qa"])
    response     = _chat(system, user)

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