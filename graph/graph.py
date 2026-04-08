from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from graph.state import DocState
from graph.nodes import (
    classify_intent,
    retriever_node,
    generate_node,
    edit_node,
    general_node,
    compare_node,
    analyse_node,
)


def route_intent(state: DocState) -> str:
    return state["intent"]


def build_graph():
    g = StateGraph(DocState)

    g.add_node("classify_intent", classify_intent)
    g.add_node("retriever",       retriever_node)
    g.add_node("generate",        generate_node)
    g.add_node("edit",            edit_node)
    g.add_node("general",         general_node)
    g.add_node("compare",         compare_node)
    g.add_node("analyse",         analyse_node)

    g.set_entry_point("classify_intent")

    g.add_conditional_edges("classify_intent", route_intent, {
        "general":   "general",
        "summarise": "retriever",
        "explain":   "retriever",
        "qa":        "retriever",
        "edit":      "edit",
        "compare":   "compare",
        "analyse":   "analyse",
    })

    g.add_edge("retriever", "generate")
    g.add_edge("generate",  END)
    g.add_edge("edit",      END)
    g.add_edge("general",   END)
    g.add_edge("compare",   END)
    g.add_edge("analyse",   END)

    return g.compile()


graph = build_graph()


def run(
    query:      str,
    session_id: str,
    memory:     list[dict] = None,
) -> dict:
    state = {
        "messages":         [HumanMessage(content=query)],
        "session_id":       session_id,
        "doc_id":           "",
        "filename":         "",
        "intent":           None,
        "retrieved_chunks": [],
        "memory":           memory or [],
        "response":         "",
        "sources":          [],
        "edit_record":      {},
        "current_text":     "",
        "edit_history":     [],
    }
    result = graph.invoke(state)
    return {
        "response":    result["response"],
        "intent":      result["intent"],
        "sources":     result.get("sources", []),
        "edit_record": result.get("edit_record", {}),
    }