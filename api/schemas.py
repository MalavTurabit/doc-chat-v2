from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    doc_id:    str
    filename:  str
    session_id: str
    chunks:    int


class ChatRequest(BaseModel):
    session_id: str
    query:      str


class ChatResponse(BaseModel):
    response:   str
    intent:     str
    sources:    list[str]   # filenames that contributed to the answer


class DownloadRequest(BaseModel):
    session_id: str


class SessionInfo(BaseModel):
    session_id:   str
    docs:         list[dict]   # [{doc_id, filename}]
    message_count: int
    edit_count:   int


class DeleteResponse(BaseModel):
    session_id: str
    deleted:    bool