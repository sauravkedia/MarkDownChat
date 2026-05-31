from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from llmrequest import OllamaRestLLM
from ingestion import VectorStoreRetriever, build_rag_pipeline
from chat_api import ChatAPI
import config

# =========================================================
# LLM INITIALIZATION
# =========================================================
llm = OllamaRestLLM(
    model=config.LLM_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=0
)

# =========================================================
# RAG CHAT APPLICATION
# =========================================================

class RAGChatApplication:

    def __init__(self, vector_store:VectorStoreRetriever):
        self.vector_store = vector_store
        self.current_collection = vector_store.index_name
        self.chat_history = []
        self.retriever = self.vector_store.get_retriever()

    def _set_collection(self, collection_name: str):
        if collection_name and collection_name != self.current_collection:
            self.vector_store._set_index_name(collection_name)
            self.retriever = self.vector_store.get_retriever()
            self.current_collection = collection_name

    def _build_prompt(self, context: str, user_query: str):
        history_text = ""

        for message in self.chat_history:
            if isinstance(message, HumanMessage):
                history_text += f"User: {message.content}\\n"
            elif isinstance(message, AIMessage):
                history_text += f"Assistant: {message.content}\\n"

        prompt = f"""
            You are a helpful and accurate Insurance AI Assistant.

            Your task is to answer user questions strictly using the provided context.

            Rules:
            1. Use ONLY the provided context.
            2. Do NOT make assumptions or generate information not present in the context.
            3. If multiple relevant details exist, summarize them clearly.
            4. Keep responses concise, professional, and easy to understand.
            5. If the context does not contain the answer, respond exactly with:
            "I could not find the answer in the product brochure."

            Chat History:
            {history_text}

            Provided Context:
            {context}

            User Question:
            {user_query}

            Answer:
            """

        return prompt

    def chat(self, user_query: str, collection_name: str | None = None):
        if collection_name:
            self._set_collection(collection_name)

        response = self.retriever.invoke(user_query)
        print("Retrieved relevant documents from vector store")

        context = "\\n\\n".join([doc.page_content for doc in response])
        prompt = self._build_prompt(context=context, user_query=user_query)
        answer = llm.invoke(prompt)

        self.chat_history.append(HumanMessage(content=user_query))
        self.chat_history.append(AIMessage(content=answer))

        return answer


# =========================================================
# CHAT INTERFACE
# =========================================================

def start_chat():
    md_file = f"{config.COLLECTION_NAME}.md"
    print(f"{md_file}")

    vector_store = build_rag_pipeline(md_file, config.COLLECTION_NAME)
    rag_app = RAGChatApplication(vector_store)

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            print("Exiting chat...")
            break

        now = datetime.now()
        print("Current date and time:", now.strftime("%Y-%m-%d %H:%M:%S"))

        response = rag_app.chat(query)

        now = datetime.now()
        print("Current date and time:", now.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"\nAI: {response}\n")



# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        print("Starting Chat Document API server (FastAPI)...")
        print("Loading vector store and initializing RAG pipeline...\n")

        md_file = f"{config.COLLECTION_NAME}.md"
        vector_store = build_rag_pipeline(md_file,config.COLLECTION_NAME)
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
        print("  curl -X POST http://localhost:5000/api/chat \\")
        print("    -H \"Content-Type: application/json\" \\")
        print(f"    -d '{{\"message\": \"What is this document about?\"}}'")
        print("=" * 70 + "\n")
        api.run()
    else:
        start_chat()
