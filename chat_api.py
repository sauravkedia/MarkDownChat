from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import *
from ingestion import ingest_markdown
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

# =========================================================
# FASTAPI APPLICATION
# =========================================================

class ChatAPI:

    def __init__(self, rag_app):
        self.rag_app = rag_app
        self.app = FastAPI(
            title="Chat Document API",
            description="RAG-based document chat API using Ollama and OpenSearch",
            version="1.0.0"
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()

    def _setup_routes(self):
        @self.app.get(
            "/api/status",
            response_model=StatusResponse,
            summary="Health check",
            tags=["Status"]
        )
        async def status():
            return StatusResponse(
                status="healthy",
                model=config.LLM_MODEL
            )

        @self.app.post(
            "/api/chat",
            response_model=ChatResponse,
            summary="Send a chat message",
            tags=["Chat"],
            responses={
                400: {"model": ErrorResponse, "description": "Invalid request"},
                500: {"model": ErrorResponse, "description": "Server error"}
            }
        )
        async def chat(chat_request: ChatRequest):
            try:
                user_message = chat_request.message.strip()
                collection_name = (chat_request.collection_name or config.COLLECTION_NAME).strip()

                if not user_message:
                    raise HTTPException(status_code=400, detail="Message cannot be empty")

                if not collection_name:
                    raise HTTPException(status_code=400, detail="Collection name cannot be empty")

                if not self.rag_app.vector_store.index_exists(collection_name):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Collection '{collection_name}' does not exist or has no documents."
                    )

                if self.rag_app.vector_store.get_document_count(collection_name) == 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Collection '{collection_name}' does not exist or has no documents."
                    )

                response = self.rag_app.chat(user_message, collection_name=collection_name)

                return ChatResponse(
                    status="success",
                    message=user_message,
                    response=response,
                    timestamp=datetime.now().isoformat()
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/api/history",
            response_model=HistoryResponse,
            summary="Get conversation history",
            tags=["History"]
        )
        async def get_history():
            try:
                history = []
                for msg in self.rag_app.chat_history:
                    if isinstance(msg, HumanMessage):
                        history.append(HistoryMessage(role="user", content=msg.content))
                    elif isinstance(msg, AIMessage):
                        history.append(HistoryMessage(role="assistant", content=msg.content))

                return HistoryResponse(status="success", history=history)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/api/reset",
            response_model=ResetResponse,
            summary="Reset conversation history",
            tags=["History"]
        )
        async def reset_history():
            try:
                self.rag_app.chat_history = []
                return ResetResponse(status="success", message="Chat history reset")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post(
            "/api/ingest",
            response_model=IngestResponse,
            summary="Ingest markdown text into OpenSearch",
            tags=["Ingestion"],
            responses={
                400: {"model": ErrorResponse, "description": "Invalid request"},
                500: {"model": ErrorResponse, "description": "Server error"}
            }
        )
        async def ingest(request: IngestRequest):
            try:
                markdown_text = request.markdown_text.strip()
                collection_name = (request.collection_name or config.COLLECTION_NAME).strip()

                if not markdown_text:
                    raise HTTPException(status_code=400, detail="Markdown text cannot be empty")

                if not collection_name:
                    raise HTTPException(status_code=400, detail="Collection name cannot be empty")

                chunk_count = ingest_markdown(markdown_text, collection_name)

                return IngestResponse(
                    status="success",
                    collection_name=collection_name,
                    chunk_count=chunk_count,
                    message=f"Added {chunk_count} chunks to OpenSearch index '{collection_name}'"
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/",
            summary="API Info",
            tags=["Info"]
        )
        async def root():
            return {
                "name": "Chat Document API",
                "version": "1.0.0",
                "description": "RAG-based document chat API",
                "docs": "/docs",
                "openapi_schema": "/openapi.json"
            }

    def run(self, host: str = "0.0.0.0", port: int = 5001, reload: bool = False):
        import uvicorn
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload
        )
