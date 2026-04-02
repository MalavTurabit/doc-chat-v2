from typing import Annotated, Literal
import operator
from langgraph.graph import MessagesState


class DocState(MessagesState):
    # document identity
    doc_id:       str
    filename:     str

    # routing
    intent: Literal["summarise", "explain", "qa", "edit"] | None

    # retrieval
    retrieved_chunks: list[dict]

    # generation
    response: str

    # edit tracking
    current_text: str
    edit_history: list[dict]