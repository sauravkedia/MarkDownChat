
from langchain_core.documents import Document

class Utility:

    @staticmethod
    def text_to_document(source: str):
        return Document(
            page_content=source,
            metadata={"source": source}
        )