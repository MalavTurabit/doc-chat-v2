from typing import Literal
from langgraph.graph import MessagesState


class DocState(MessagesState):
    session_id:       str
    doc_id:           str
    filename:         str
    intent:           Literal["general","summarise","explain","qa","edit","compare","analyse"] | None
    query_type:       str      
    retrieved_chunks: list[dict]
    memory:           list[dict]
    response:         str
    sources:          list[str]
    edit_record:      dict
    current_text:     str
    edit_history:     list[dict]