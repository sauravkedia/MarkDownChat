# =========================================================
# CONFIG
# =========================================================

OLLAMA_BASE_URL = "http://localhost:11434"

LLM_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "nomic-embed-text"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "optima_secure"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

TOP_K = 5


VECTOR_DB_PATH = "vector_store"