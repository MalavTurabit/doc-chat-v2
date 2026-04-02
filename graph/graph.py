from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from graph.state import DocState
from graph.nodes import (
    classify_intent,
    retriever_node,
    generate_node,
    edit_node,
)


def route_intent(state: DocState) -> str:
    return state["intent"]


def build_graph():
    g = StateGraph(DocState)

    g.add_node("classify_intent", classify_intent)
    g.add_node("retriever",       retriever_node)
    g.add_node("generate",        generate_node)
    g.add_node("edit",            edit_node)

    g.set_entry_point("classify_intent")

    g.add_conditional_edges("classify_intent", route_intent, {
        "summarise": "retriever",
        "explain":   "retriever",
        "qa":        "retriever",
        "edit":      "edit",
    })

    g.add_edge("retriever", "generate")
    g.add_edge("generate",  END)
    g.add_edge("edit",      END)

    return g.compile()


graph = build_graph()


def run(query: str, doc_id: str, filename: str,
        current_text: str = "", edit_history: list = None) -> dict:
    """
    Returns a dict with response, edit_history, current_text.
    """
    state = {
        "messages":         [HumanMessage(content=query)],
        "doc_id":           doc_id,
        "filename":         filename,
        "intent":           None,
        "retrieved_chunks": [],
        "response":         "",
        "current_text":     current_text,
        "edit_history":     edit_history or [],
    }
    result = graph.invoke(state)
    return {
        "response":     result["response"],
        "edit_history": result.get("edit_history", []),
        "current_text": result.get("current_text", ""),
    }