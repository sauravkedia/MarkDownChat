
from pydantic import BaseModel
from typing import Optional
import config

# =========================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE VALIDATION
# =========================================================

class ChatRequest(BaseModel):
    message: str
    collection_name: Optional[str] = config.COLLECTION_NAME

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is this document about?",
                "collection_name": "insurance_docs"
            }
        }


class ChatResponse(BaseModel):
    status: str
    message: str
    response: str
    timestamp: str


class IngestRequest(BaseModel):
    markdown_text: str
    collection_name: Optional[str] = config.COLLECTION_NAME

    class Config:
        json_schema_extra = {
            "example": {
                "markdown_text": "# Hello\\nThis is sample markdown content.",
                "collection_name": "insurance_docs"
            }
        }


class IngestResponse(BaseModel):
    status: str
    collection_name: str
    chunk_count: int
    message: str


class StatusResponse(BaseModel):
    status: str
    model: str
    collection: str


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    status: str
    history: list[HistoryMessage]


class ResetResponse(BaseModel):
    status: str
    message: str


class ErrorResponse(BaseModel):
    error: str
    status: str

