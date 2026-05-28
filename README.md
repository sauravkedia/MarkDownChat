# Chat Document - RAG-Based Chat Application

A production-ready **Retrieval-Augmented Generation (RAG)** chat application that combines Ollama LLM with OpenSearch for intelligent document-based conversations.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Features

✅ **FastAPI Backend** - Modern, fast, and production-ready REST API
✅ **Interactive Documentation** - Swagger UI and ReDoc available at `/docs`
✅ **Type Validation** - Request/response validation with Pydantic
✅ **Async Support** - Full async/await support for scalability
✅ **CORS Enabled** - Cross-origin resource sharing
✅ **RAG Pipeline** - Retrieval-Augmented Generation with context awareness
✅ **Conversation History** - Maintains chat context for multi-turn conversations
✅ **Dual Mode** - Run as CLI or REST API server
✅ **Search Engine** - OpenSearch for powerful full-text and semantic search
✅ **Local LLM** - Ollama-powered inference

---

## Architecture

```
┌─────────────────┐
│   Client (CLI)  │
└────────┬────────┘
         │
    ┌────▼──────────┐
    │  Application  │
    └────┬──────────┘
         │
    ┌────▼──────────────────────────────┐
    │   RAG Chat Application            │
    │  ┌──────────────┐  ┌───────────┐  │
    │  │  Retriever   │  │    LLM    │  │
    │  └──────┬───────┘  └─────▲─────┘  │
    │         │                │        │
    │  ┌──────▼────────┐ ┌─────┴─────┐  │
    │  │ OpenSearch DB │ │   Ollama  │  │
    │  └──────────────┘ └───────────┘  │
    └───────────────────────────────────┘
         │
         │ (FastAPI + Uvicorn)
         │
    ┌────▼──────────────┐
    │  HTTP REST API    │
    │  :5000            │
    └───────────────────┘
```

---

## Prerequisites

### Required Services

1. **Ollama** - Local LLM inference
   ```bash
   # Install from: https://ollama.ai
   ollama serve
   ```

2. **OpenSearch** - Distributed search and analytics engine
   ```bash
   docker run -d --name opensearch -p 9200:9200 -p 9600:9600 \
     -e "discovery.type=single-node" \
     -e "plugins.security.disabled=true" \
     -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=Insurance@2026" \
     opensearchproject/opensearch:latest
   ```

### Models

Download required models:
```bash
ollama pull qwen2.5        # Main LLM model
ollama pull bge-m3:567m    # Embedding model
```

---

## Installation

### 1. Clone/Setup Repository

```bash
cd /path/to/DocumentChat
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `langchain` - LLM framework
- `langchain-community` - Community integrations
- `langchain-ollama` - Ollama integration
- `langchain-experimental` - Semantic chunking utilities
- `opensearch-py` - OpenSearch client
- `requests-aws4auth` - AWS authentication for OpenSearch
- `fastapi` - REST API framework
- `uvicorn[standard]` - ASGI server

### 3. Prepare Documents

Place your markdown document in the repository root:
```bash
# Default document: optima_secure.md
# Update COLLECTION_NAME in chat_document.py to use different documents
```

---

## Running the Application

### CLI Mode (Interactive Chat)

```bash
python chat_document.py
```

You'll see an interactive prompt where you can chat with the AI:
```
You: What are the main features?
Current date and time: 2026-05-28 10:30:45

AI: [Response based on document context]

You: exit
```

### API Mode (FastAPI Server)

```bash
python chat_document.py --api
```

Output:
```
Starting Chat Document API server (FastAPI)...
Loading vector store and initializing RAG pipeline...

✅ API server ready!
======================================================================
🚀 FastAPI server running on http://0.0.0.0:5000

📚 Interactive API Documentation:
  Swagger UI: http://localhost:5000/docs
  ReDoc:      http://localhost:5000/redoc

📡 Available endpoints:
  GET  /api/status     - Check API status
  POST /api/chat       - Send a chat message
  GET  /api/history    - Get conversation history
  POST /api/reset      - Reset conversation history

💡 Example request:
  curl -X POST http://localhost:5000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "What is this document about?"}'
======================================================================
```

---

## API Documentation

### Access Interactive Documentation

Once the API is running:
- **Swagger UI (Recommended)**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc

You can test all endpoints directly in the browser!

### API Endpoints

#### 1. Health Check
**Endpoint:** `GET /api/status`

Check if the API server is running and get configuration info.

**Response:**
```json
{
  "status": "healthy",
  "model": "qwen2.5",
  "collection": "optima_secure"
}
```

**cURL:**
```bash
curl http://localhost:5000/api/status
```

---

#### 2. Send Chat Message
**Endpoint:** `POST /api/chat`

Send a user message and receive an AI response based on document context.

**Request Body:**
```json
{
  "message": "Your question here"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Your question here",
  "response": "AI response based on document context",
  "timestamp": "2026-05-28T10:30:45.123456"
}
```

**cURL:**
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the main features?"}'
```

---

#### 3. Get Conversation History
**Endpoint:** `GET /api/history`

Retrieve the complete conversation history with all user messages and AI responses.

**Response:**
```json
{
  "status": "success",
  "history": [
    {
      "role": "user",
      "content": "First question"
    },
    {
      "role": "assistant",
      "content": "First answer"
    },
    {
      "role": "user",
      "content": "Second question"
    },
    {
      "role": "assistant",
      "content": "Second answer"
    }
  ]
}
```

**cURL:**
```bash
curl http://localhost:5000/api/history
```

---

#### 4. Reset Conversation History
**Endpoint:** `POST /api/reset`

Clear the conversation history and start fresh.

**Response:**
```json
{
  "status": "success",
  "message": "Chat history reset"
}
```

**cURL:**
```bash
curl -X POST http://localhost:5000/api/reset
```

---

## Usage Examples

### Python Integration

Create a file `chat_client.py`:

```python
import requests
import json

class ChatClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def chat(self, message):
        """Send a chat message and get response"""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"message": message}
        )
        return response.json()
    
    def get_history(self):
        """Get conversation history"""
        response = requests.get(f"{self.base_url}/api/history")
        return response.json()
    
    def reset(self):
        """Reset conversation history"""
        response = requests.post(f"{self.base_url}/api/reset")
        return response.json()
    
    def status(self):
        """Check API status"""
        response = requests.get(f"{self.base_url}/api/status")
        return response.json()

# Usage
if __name__ == "__main__":
    client = ChatClient()
    
    # Check status
    print("Status:", client.status())
    
    # Send a message
    result = client.chat("Tell me about this document")
    print("Response:", result["response"])
    
    # Get history
    history_data = client.get_history()
    for msg in history_data["history"]:
        print(f"{msg['role'].upper()}: {msg['content'][:100]}...")
```

### JavaScript/Node.js Integration

```javascript
const BASE_URL = "http://localhost:5000";

// Check status
fetch(`${BASE_URL}/api/status`)
  .then(res => res.json())
  .then(data => console.log("Status:", data));

// Send a chat message
const message = "What is this document about?";
fetch(`${BASE_URL}/api/chat`, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({message: message})
})
  .then(res => res.json())
  .then(data => console.log("Response:", data.response));

// Get conversation history
fetch(`${BASE_URL}/api/history`)
  .then(res => res.json())
  .then(data => {
    data.history.forEach(msg => {
      console.log(`${msg.role}: ${msg.content}`);
    });
  });

// Reset history
fetch(`${BASE_URL}/api/reset`, {method: "POST"})
  .then(res => res.json())
  .then(data => console.log(data.message));
```

### cURL Examples

```bash
# Check status
curl http://localhost:5000/api/status

# Send a chat message
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the main features?"}'

# Get conversation history
curl http://localhost:5000/api/history

# Reset history
curl -X POST http://localhost:5000/api/reset
```

---

## Configuration

Edit `config.py` to modify settings:

```python
# LLM Configuration
OLLAMA_MODEL = "qwen2.5"              # LLM model name
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama server URL

# Embedding Configuration
EMBEDDING_MODEL = "bge-m3:567m"       # Embedding model name
EMBEDDING_DIMENSION = 1024            # Embedding vector dimension

# OpenSearch Configuration
OPENSEARCH_HOST = "localhost"          # OpenSearch host
OPENSEARCH_PORT = 9200                # OpenSearch port
OPENSEARCH_INDEX = "optima_secure"    # Index name

# Retrieval Configuration
TOP_K = 10                            # Number of relevant documents to retrieve
```

---

## Error Handling

The API returns appropriate HTTP status codes:

- `200 OK` - Successful request
- `400 Bad Request` - Missing or invalid request parameters
- `500 Internal Server Error` - Server error or exception

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Performance Tips

1. **First Load:** The first chat request may take longer as OpenSearch indexes are initialized.

2. **Context Window:** The API maintains chat history for context-aware responses. Reset history if needed for long sessions:
   ```bash
   curl -X POST http://localhost:5000/api/reset
   ```

3. **Timeout:** Long documents may take time to process. The system has a 300-second timeout for LLM requests.

4. **Optimization:** 
   - Reduce `TOP_K` value for faster retrieval
   - Tune OpenSearch refresh interval for indexing performance
   - Use semantic chunking for better document relevance
   - Optimize embedding model size for your hardware constraints

5. **OpenSearch Tuning:**
   - Check index health: `curl http://localhost:9200/_cluster/health`
   - Monitor indexing: `curl http://localhost:9200/optima_secure/_stats`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ConnectionRefusedError: Ollama` | Ensure Ollama is running: `ollama serve` |
| `Connection error to OpenSearch` | Ensure OpenSearch is running: `docker run -d --name opensearch -p 9200:9200 -e "discovery.type=single-node" -e "plugins.security.disabled=true" -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=Insurance@2026" opensearchproject/opensearch:latest` |
| `FileNotFoundError: markdown file` | Place your markdown document in the working directory (default: `optima_secure.md`) |
| `Empty or slow responses` | Reduce `TOP_K` value or ensure document is indexed correctly in OpenSearch |
| `API docs not loading` | Visit http://localhost:5000/docs (note the `/docs` path) |
| `Port already in use` | Change port in code: `api.run(port=8000)` |
| `OpenSearch index not found` | Ensure the index is created and documents are indexed in OpenSearch |

---

## Advanced Deployment

### Production with Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker chat_document:app --bind 0.0.0.0:5000
```

### Docker Deployment

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "chat_document.py", "--api"]
```

Build and run:
```bash
docker build -t chat-document .
docker run -p 5000:5000 chat-document
```

---

## Project Structure

```
DocumentChat/
├── chat_document.py           # Main application
├── config.py                  # Configuration settings
├── llmrequest.py             # Ollama LLM client
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── Dockerfile                # Docker configuration
├── optima_secure.md          # Sample document
└── ergo_hca.md              # Sample document
```

---

## Key Components

### ChatAPI (FastAPI)
Handles HTTP REST endpoints with automatic validation and documentation.

### RAGChatApplication
Manages the retrieval-augmented generation pipeline and chat history.

### VectorStoreManager
Manages OpenSearch index and document indexing operations.

### MarkdownLoader
Loads and processes markdown documents.

### SemanticDocumentChunker
Splits documents into semantic chunks using semantic analysis.

### OllamaRestLLM
Communicates with Ollama for LLM inference.

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review API documentation at http://localhost:5000/docs
3. Check application logs for detailed error messages

---

## License

This project is proprietary and confidential.

---

**Last Updated:** 2026-05-28
**Version:** 1.0.0
