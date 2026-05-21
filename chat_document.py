"""
Production Grade Markdown RAG Application
=========================================

Start Qdrant:
-------------
docker run -p 6333:6333 qdrant/qdrant

Start Ollama:
-------------
ollama pull mistral
ollama pull nomic-embed-text

Run:
----
python chat_document.py
"""

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import ExperimentalMarkdownSyntaxTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


from langchain_ollama import (
    OllamaLLM,
    OllamaEmbeddings
)

from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# =========================================================
# CONFIGURATION
# =========================================================

OLLAMA_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"
VECTOR_DB_PATH = "vector_store"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "optima_secure"
TOP_K = 15
# =========================================================
# LLM INITIALIZATION
# =========================================================

llm = OllamaLLM(
    model=OLLAMA_MODEL,
    temperature=0
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
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

class MDDocumentChunker:

    @staticmethod
    def chunk_documents(documents):
        """
        Split documents into chunks
        """

        markdown_text = "\n".join(
            [doc.page_content for doc in documents]
        )

        headers_to_split_on = [
            ("#", "Header 1"),
            # ("##", "Header 2"),
            # ("###", "Header 3"),
        ]

        # Step 3: Initialize markdown splitter
        markdown_splitter = (
            ExperimentalMarkdownSyntaxTextSplitter(
                headers_to_split_on = headers_to_split_on,
                strip_headers=False
            )
        )

        # markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

        # Step 4: Split markdown
        split_docs = markdown_splitter.split_text(
            markdown_text
        )
        return split_docs

# =========================================================
# VECTOR STORE MANAGEMENT
# =========================================================

class VectorStoreManager:

    def __init__(self, chunks):
        """
        Create QDrant vector store
        """

        self.qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT
        )

        self._create_collection_if_not_exists()

        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings
        )

        self.add_documents(chunks)

    
    def _create_collection_if_not_exists(self):

        collections = self.qdrant_client.get_collections()

        existing_collections = [
            c.name for c in collections.collections
        ]

        if COLLECTION_NAME not in existing_collections:

            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            )

            print(f"Created collection: {COLLECTION_NAME}")

    def add_documents(
        self,
        documents: list[Document]
    ):

        self.vector_store.add_documents(documents)

        print(f"Stored {len(documents)} chunks in vector DB")

    def get_retriever(self):

        return self.vector_store.as_retriever(
            search_kwargs={
                "k": TOP_K
            }
        )
    

# =========================================================
# RAG CHAT APPLICATION
# =========================================================

class RAGChatApplication:

    def __init__(self, vector_store):

        self.vector_store = vector_store

        self.chat_history = []

        self.retriever = self.vector_store.get_retriever()

        self.rag_chain = self._build_rag_chain()

    def _build_rag_chain(self):
        """
        Build retrieval chain using create_retrieval_chain
        """

        system_prompt = """
            You are a helpful and accurate Insurance AI Assistant.

            Your task is to answer user questions strictly using the provided context.

            Rules:
            1. Use ONLY the provided context.
            2. Do NOT make assumptions or generate information not present in the context.
            3. If multiple relevant details exist, summarize them clearly.
            4. Keep responses concise, professional, and easy to understand.
            5. If the context does not contain the answer, respond exactly with:
            "I could not find the answer in the provided documents."

            Provided Context:
            {context}
            """

        prompt = ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),("human","{input}")
            ]
)
        # print("\n\n---- prompt created start ----\n\n")
        # print(f"{prompt}")
        # print("\n\n---- prompt created end ----\n\n")
        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt
        )

        # print("\n\n---- document_chain start ----\n\n")
        # print(f"{document_chain}")
        # print("\n\n---- document_chain end ----\n\n")

        retrieval_chain = create_retrieval_chain(
            self.retriever,
            document_chain
        )

        # print("\n\n---- retrieval_chain start ----\n\n")
        # print(f"retrieval_chain: {retrieval_chain}")
        # print("\n\n---- retrieval_chain end ----\n\n")

        return retrieval_chain

    def chat(self, user_query: str):
        """
        Chat with RAG application
        """

        print(f"\nUser query: {user_query}\n")
        print(f"Chat history: {self.chat_history}\n")

        response = self.rag_chain.invoke(
            {
                "input": user_query,
                "chat_history": self.chat_history
            }
        )

        answer = response["answer"]

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
    chunks = MDDocumentChunker.chunk_documents(
        documents
    )

    # Step 3: Create vector store
    vector_store = VectorStoreManager(chunks)

    return vector_store


# =========================================================
# CHAT INTERFACE
# =========================================================

def start_chat():

    md_file = "optima_secure.md"

    # Build vector DB
    vector_store = build_rag_pipeline(md_file)

    rag_app = RAGChatApplication(vector_store)

    # print("\n==============================")
    # print("Markdown RAG Chat Started")
    # print("==============================")
    # print("Type 'exit' to stop\n")

    while True:

        query = input("You: ")

        if query.lower() == "exit":
            print("Exiting chat...")
            break

        response = rag_app.chat(query)

        print(f"\nAI: {response}\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # Start chat
    start_chat()