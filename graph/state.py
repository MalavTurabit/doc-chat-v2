from typing import Literal
from langgraph.graph import MessagesState


class DocState(MessagesState):
    # session
    session_id: str

    # document (optional — used only for single-doc edit scoping)
    doc_id:   str
    filename: str

    # routing
    intent: Literal["summarise", "explain", "qa", "edit"] | None

    # retrieval
    retrieved_chunks: list[dict]

    # memory — last N messages injected into prompts
    memory: list[dict]

    # generation
    response:    str
    sources:     list[str]
    edit_record: dict

    # edit tracking
    current_text: str
    edit_history: list[dict]