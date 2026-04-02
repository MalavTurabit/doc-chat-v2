from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_LLM_KEY,
    AZURE_LLM_ENDPOINT,
    AZURE_LLM_API_VERSION,
    AZURE_LLM_DEPLOYMENT,
)
from vectorstore.embedder import embed_query
from vectorstore.milvus_client import search, get_all_chunks
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
        intent = "qa"   # safe default

    print(f"[intent] → {intent}")
    return {"intent": intent}


# ── Retriever ─────────────────────────────────────────────────────────────────

def retriever_node(state: DocState) -> dict:
    query    = state["messages"][-1].content
    doc_id   = state["doc_id"]
    intent   = state["intent"]

    if intent == "summarise":
        chunks = get_all_chunks(doc_id)
    else:
        q_vec  = embed_query(query)
        chunks = search(q_vec, doc_id, top_k=5)

    print(f"[retriever] fetched {len(chunks)} chunks  (intent={intent})")
    return {"retrieved_chunks": chunks}


# ── Generate ──────────────────────────────────────────────────────────────────

def generate_node(state: DocState) -> dict:
    query    = state["messages"][-1].content
    intent   = state["intent"]
    chunks   = state["retrieved_chunks"]

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


# ── Edit (stub for Module 4) ──────────────────────────────────────────────────

def edit_node(state: DocState) -> dict:
    return {"response": "Edit pipeline coming in Module 4."}