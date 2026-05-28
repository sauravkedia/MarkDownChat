from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

# from langchain_text_splitters import ExperimentalMarkdownSyntaxTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch

from datetime import datetime

from langchain_ollama import (
    OllamaEmbeddings
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from llmrequest import OllamaRestLLM
import config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# =========================================================
# LLM INITIALIZATION
# =========================================================
llm = OllamaRestLLM(
    model=config.LLM_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=0
)

embeddings = OllamaEmbeddings(
    model=config.EMBEDDING_MODEL,
    base_url=config.OLLAMA_BASE_URL
)

# =========================================================
# DOCUMENT LOADER
# =========================================================

class MarkdownLoader:

    @staticmethod
    def load_markdown(md_file_path: str):
        """
        Load markdown file
        """

        loader = TextLoader(
            md_file_path,
            encoding="utf-8"
        )

        documents = loader.load()

        # print(f"Loaded markdown file: {md_file_path}")

        return documents


# =========================================================
# DOCUMENT CHUNKING
# =========================================================

class SemanticDocumentChunker:

    @staticmethod
    def chunk_documents(documents):
        """
        Split documents into chunks
        """

        markdown_text = [doc.page_content for doc in documents]

        semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",  # or "standard_deviation"
            breakpoint_threshold_amount=95           # higher = bigger chunks
        )

        split_docs = semantic_chunker.create_documents(markdown_text)

        return split_docs

# class MDDocumentChunker:

#     @staticmethod
#     def chunk_documents(documents):
#         """
#         Split documents into chunks
#         """

#         markdown_text = "\n".join(
#             [doc.page_content for doc in documents]
#         )

#         headers_to_split_on = [
#             ("#", "Header_1"),
#             # ("##", "Header_2"),
#             # ("###", "Header_3"),
#         ]

#         # Step 3: Initialize markdown splitter
#         markdown_splitter = (
#             ExperimentalMarkdownSyntaxTextSplitter(
#                 headers_to_split_on = headers_to_split_on,
#                 strip_headers=False
#             )
#         )


#         # Step 4: Split markdown
#         split_docs = markdown_splitter.split_text(
#             markdown_text
#         )
#         return split_docs

# =========================================================
# VECTOR STORE MANAGEMENT
# =========================================================
    
class VectorStoreManager:

    def __init__(self, chunks):

        self.index_name = config.OPENSEARCH_INDEX

        self.client = OpenSearch(
            hosts=[{
                "host": config.OPENSEARCH_HOST,
                "port": config.OPENSEARCH_PORT
            }],
            http_compress=True,
            use_ssl=False,
            verify_certs=False
        )

        self._create_index_if_not_exists()

        self.vector_store = OpenSearchVectorSearch(
            index_name=self.index_name,
            embedding_function=embeddings,
            opensearch_url=f"http://{config.OPENSEARCH_HOST}:{config.OPENSEARCH_PORT}"
        )

        # Add documents only if index is empty
        count = self.client.count(index=self.index_name)["count"]

        if count == 0:
            self.add_documents(chunks)

    def _create_index_if_not_exists(self):

        if not self.client.indices.exists(index=self.index_name):

            index_body = {
                "settings": {
                    "index": {
                        "knn": True
                    }
                },
                "mappings": {
                    "properties": {
                        "vector_field": {
                            "type": "knn_vector",
                            "dimension": config.EMBEDDING_DIMENSION,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene"
                            }
                        },
                        "text": {
                            "type": "text"
                        }
                    }
                }
            }

            self.client.indices.create(
                index=self.index_name,
                body=index_body
            )

            print(f"Created OpenSearch index: {self.index_name}")

    def add_documents(self, documents):

        self.vector_store.add_documents(documents)

        print(f"Stored {len(documents)} chunks in OpenSearch")

    def get_retriever(self):

        return self.vector_store.as_retriever(
            search_kwargs={"k": config.TOP_K}
        )
    
# =========================================================
# RAG CHAT APPLICATION
# =========================================================

class RAGChatApplication:

    def __init__(self, vector_store):

        self.vector_store = vector_store

        self.chat_history = []

        self.retriever = self.vector_store.get_retriever()

    
    def _build_prompt(self, context: str, user_query: str):

        history_text = ""

        for message in self.chat_history:

            if isinstance(message, HumanMessage):
                history_text += f"User: {message.content}\n"

            elif isinstance(message, AIMessage):
                history_text += f"Assistant: {message.content}\n"

        prompt = f"""
            You are a helpful and accurate Insurance AI Assistant.

            Your task is to answer user questions strictly using the provided context.

            Rules:
            1. Use ONLY the provided context.
            2. Do NOT make assumptions or generate information not present in the context.
            3. If multiple relevant details exist, summarize them clearly.
            4. Keep responses concise, professional, and easy to understand.
            5. If the context does not contain the answer, respond exactly with:
            "I could not find the answer in the provided documents."

            Chat History:
            {history_text}

            Provided Context:
            {context}

            User Question:
            {user_query}

            Answer:
            """

        return prompt

    def chat(self, user_query: str):
        """
        Chat with RAG application
        """

        # Step 1: Retrieve relevant documents
        response = self.retriever.invoke(user_query)
        print("Retrieved relevant documents from vector store")
        # print(f"Retrieved relevant documents from vector store: {response}")
        
        # Step 2: Build context
        context = "\n\n".join(
            [doc.page_content for doc in response]
        )

        # Step 3: Create prompt
        prompt = self._build_prompt(
            context=context,
            user_query=user_query
        )

        # Step 4: Call Ollama REST API
        answer = llm.invoke(prompt)
        # answer = llm.invoke(prompt)

        # Save chat history
        self.chat_history.append(
            HumanMessage(content=user_query)
        )

        self.chat_history.append(
            AIMessage(content=answer)
        )

        return answer


# =========================================================
# BUILD RAG PIPELINE
# =========================================================

def build_rag_pipeline(md_file_path: str):
    """
    End-to-end RAG pipeline
    """

    # Step 1: Load markdown
    documents = MarkdownLoader.load_markdown(
        md_file_path
    )

    # Step 2: Chunk documents
    chunks = SemanticDocumentChunker.chunk_documents(
        documents
    )

    # Step 3: Create vector store
    vector_store = VectorStoreManager(chunks)

    return vector_store


# =========================================================
# CHAT INTERFACE
# =========================================================

def start_chat():

    md_file = f"{config.COLLECTION_NAME}.md"

    print(f"{md_file}")

    # Build vector DB
    vector_store = build_rag_pipeline(md_file)

    rag_app = RAGChatApplication(vector_store)

    while True:

        query = input("You: ")

        if query.lower() == "exit":
            print("Exiting chat...")
            break

        now = datetime.now()
        formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        print("Current date and time:", formatted)
        
        response = rag_app.chat(query)

        now = datetime.now()
        formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        print("Current date and time:", formatted)
        print(f"\nAI: {response}\n")


# =========================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE VALIDATION
# =========================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {"message": "What is this document about?"}
        }

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    status: str
    message: str
    response: str
    timestamp: str

class StatusResponse(BaseModel):
    """Response model for status endpoint"""
    status: str
    model: str
    collection: str

class HistoryMessage(BaseModel):
    """Model for individual history message"""
    role: str
    content: str

class HistoryResponse(BaseModel):
    """Response model for history endpoint"""
    status: str
    history: list[HistoryMessage]

class ResetResponse(BaseModel):
    """Response model for reset endpoint"""
    status: str
    message: str

class ErrorResponse(BaseModel):
    """Response model for error responses"""
    error: str
    status: str


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
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get(
            "/api/status",
            response_model=StatusResponse,
            summary="Health check",
            tags=["Status"]
        )
        async def status():
            """Check if the API server is running and get configuration info."""
            return StatusResponse(
                status="healthy",
                model=config.LLM_MODEL,
                collection=config.COLLECTION_NAME
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
            """Send a user message and receive an AI response based on document context."""
            try:
                user_message = chat_request.message.strip()
                
                if not user_message:
                    raise HTTPException(
                        status_code=400,
                        detail="Message cannot be empty"
                    )
                
                # Get response from RAG application
                response = self.rag_app.chat(user_message)
                
                return ChatResponse(
                    status="success",
                    message=user_message,
                    response=response,
                    timestamp=datetime.now().isoformat()
                )
            
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=str(e)
                )
        
        @self.app.get(
            "/api/history",
            response_model=HistoryResponse,
            summary="Get conversation history",
            tags=["History"]
        )
        async def get_history():
            """Retrieve the complete conversation history."""
            try:
                history = []
                for msg in self.rag_app.chat_history:
                    if isinstance(msg, HumanMessage):
                        history.append(
                            HistoryMessage(role="user", content=msg.content)
                        )
                    elif isinstance(msg, AIMessage):
                        history.append(
                            HistoryMessage(role="assistant", content=msg.content)
                        )
                
                return HistoryResponse(
                    status="success",
                    history=history
                )
            
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=str(e)
                )
        
        @self.app.post(
            "/api/reset",
            response_model=ResetResponse,
            summary="Reset conversation history",
            tags=["History"]
        )
        async def reset_history():
            """Clear the conversation history and start fresh."""
            try:
                self.rag_app.chat_history = []
                return ResetResponse(
                    status="success",
                    message="Chat history reset"
                )
            
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=str(e)
                )
        
        @self.app.get(
            "/",
            summary="API Info",
            tags=["Info"]
        )
        async def root():
            """Get API information and documentation links."""
            return {
                "name": "Chat Document API",
                "version": "1.0.0",
                "description": "RAG-based document chat API",
                "docs": "/docs",
                "openapi_schema": "/openapi.json"
            }
    
    def run(self, host: str = '0.0.0.0', port: int = 5001, reload: bool = False):
        """Start the FastAPI server using Uvicorn"""
        import uvicorn
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--api':
        # Start API mode
        print("Starting Chat Document API server (FastAPI)...")
        print("Loading vector store and initializing RAG pipeline...\n")
        
        md_file = f"{config.COLLECTION_NAME}.md"
        vector_store = build_rag_pipeline(md_file)
        rag_app = RAGChatApplication(vector_store)
        
        api = ChatAPI(rag_app)
        print("\n✅ API server ready!")
        print("=" * 70)
        print(f"🚀 FastAPI server running on http://0.0.0.0:5000")
        print(f"\n📚 Interactive API Documentation:")
        print(f"  Swagger UI: http://localhost:5000/docs")
        print(f"  ReDoc:      http://localhost:5000/redoc")
        print(f"\n📡 Available endpoints:")
        print(f"  GET  /api/status     - Check API status")
        print(f"  POST /api/chat       - Send a chat message")
        print(f"  GET  /api/history    - Get conversation history")
        print(f"  POST /api/reset      - Reset conversation history")
        print(f"\n💡 Example request:")
        print(f'  curl -X POST http://localhost:5000/api/chat \\')
        print(f'    -H "Content-Type: application/json" \\')
        print(f'    -d \'{{"message": "What is this document about?"}}\'')
        print("=" * 70 + "\n")
        api.run()
    else:
        # Start CLI mode (default)
        start_chat()