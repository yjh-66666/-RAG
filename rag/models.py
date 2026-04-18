from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    request_id: str


class IngestTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    progress: int = 0
    detail: str = ""
    result: Dict[str, Any] = Field(default_factory=dict)


class DocumentSource(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str
    file_name: str
    ext: str
    file_size: int
    file_hash: str
    uploaded_by: str = "system"
    upload_time: datetime = Field(default_factory=datetime.now)
    last_modified: datetime
    department: str
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    department: str
    k: int = 10
    expr: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    user: str
    department: str
    session_id: str = "default"
    expr: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
