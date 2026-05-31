import requests
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from langchain_ollama import OllamaEmbeddings

import config
import utility

embeddings = OllamaEmbeddings(
    model=config.EMBEDDING_MODEL,
    base_url=config.OLLAMA_BASE_URL
)


class MarkdownLoader:

    @staticmethod
    def load_markdown(source: str):
        """
        Load markdown content from either:
        - Local file path
        - Remote URL
        """

        if source.startswith("http://") or source.startswith("https://"):
            response = requests.get(source)
            response.raise_for_status()

            markdown_content = response.text
            return [utility.text_to_document(markdown_content)]

        loader = TextLoader(source, encoding="utf-8")
        return loader.load()


# Optimize to take string as input and avoid unnecessary file type conversions
class SemanticDocumentChunker:

    @staticmethod
    def chunk_documents(documents: Document):
        """
        Split documents into chunks
        """

        markdown_text = [doc.page_content for doc in documents]

        semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )

        return semantic_chunker.create_documents(markdown_text)


class OpenSearchClient:

    def __init__(self):
        self.client = OpenSearch(
            hosts=[{
                "host": config.OPENSEARCH_HOST,
                "port": config.OPENSEARCH_PORT
            }],
            http_compress=True,
            use_ssl=False,
            verify_certs=False
        )

    def get_client(self):
        return self.client


class VectorStoreFactory:

    @staticmethod
    def create(index_name: str):

        return OpenSearchVectorSearch(
            index_name=index_name,
            embedding_function=embeddings,
            opensearch_url=(
                f"http://{config.OPENSEARCH_HOST}:"
                f"{config.OPENSEARCH_PORT}"
            )
        )


class VectorStoreRetriever:

    def __init__(self, index_name: str):

        self.index_name = index_name

        self.vector_store_factory = VectorStoreFactory.create(
            index_name=index_name
        )

    def _set_index_name(self,index_name: str):
        self.index_name = index_name
        self.vector_store_factory = VectorStoreFactory.create(index_name=index_name)

    def get_retriever(self, k: int = None):

        return self.vector_store_factory.as_retriever(
            search_kwargs={
                "k": k or config.TOP_K
            }
        )


class IndexManager:

    def __init__(self, client: OpenSearch, index_name: str):
        self.client = client
        self.index_name = index_name

    def exists(self) -> bool:
        return self.client.indices.exists(index=self.index_name)

    def create_if_not_exists(self):

        if self.exists():
            return

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

        print(f"Created index: {self.index_name}")

    def document_count(self) -> int:

        if not self.exists():
            return 0

        return self.client.count(
            index=self.index_name
        )["count"]


class DocumentIngestor:

    def __init__(self, index_name: str):

        self.index_name = index_name

        opensearch_client = OpenSearchClient().get_client()

        self.index_manager = IndexManager(
            client=opensearch_client,
            index_name=index_name
        )

        self.index_manager.create_if_not_exists()

        self.vector_store = VectorStoreFactory.create(
            index_name=index_name
        )

    def ingest(self, documents):

        if not documents:
            print("No documents to ingest")
            return

        self.vector_store.add_documents(documents)

        print(
            f"Ingested {len(documents)} documents "
            f"into index: {self.index_name}"
        )

    def ingest_if_empty(self, documents):

        count = self.index_manager.document_count()

        if count == 0:
            self.ingest(documents)
        else:
            print(
                f"Index {self.index_name} already "
                f"contains {count} documents"
            )


def ingest_markdown(markdown_text: str, collection_name: str):
    """Chunk markdown content and store it in the named OpenSearch index."""
    documents = [utility.text_to_document(markdown_text)]
    chunks = SemanticDocumentChunker.chunk_documents(documents)

    ingestor = DocumentIngestor(
        index_name=collection_name
    )

    ingestor.ingest_if_empty(chunks)
    return len(chunks)


def build_rag_pipeline(md_file_path: str, collection_name: str | None = None):
    """
    End-to-end RAG pipeline
    """
    documents = MarkdownLoader.load_markdown(md_file_path)
    chunks = SemanticDocumentChunker.chunk_documents(documents)

    ingestor = DocumentIngestor(
        index_name=collection_name
    )

    ingestor.ingest_if_empty(chunks)
    return VectorStoreRetriever(index_name=collection_name)
