import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
PARENT_STORE_DIR = os.path.join("chroma_diem", "parent_store")

print(f"Ollama chat model: {OLLAMA_CHAT_MODEL}")

# ─────────────────────────────────────────────────────────────────────────────
# Shared embedding model (used by both ingestion and app)
# ─────────────────────────────────────────────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    encode_kwargs={"normalize_embeddings": True}
)


# ─────────────────────────────────────────────────────────────────────────────
# DiemBrain — LLM, prompts, and RAG chain
# ─────────────────────────────────────────────────────────────────────────────
class DiemBrain:
    def __init__(self, vectorstore: Chroma):
        self.chat_model = ChatOllama(
            model=OLLAMA_CHAT_MODEL,
            temperature=0.1,
        )

        parent_docstore = create_kv_docstore(LocalFileStore(PARENT_STORE_DIR))
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=parent_docstore,
            child_splitter=child_splitter,
            search_kwargs={"k": 5},
        )

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant for the DIEM department "
             "(Department of Information and Electrical Engineering and Applied Mathematics) "
             "at the University of Salerno, Italy. "
             "Answer questions using ONLY the provided context. Do not use prior knowledge. "
             "If the answer is not in the context, say: "
             "'I don't have that information in my knowledge base.' "
             "If the question is unrelated to DIEM or the University of Salerno, say: "
             "'This question is outside my scope. I can only answer questions about DIEM.'"
             ),
            ("placeholder", "{history}"),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"),
        ])

        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Rewrite the user's question into a self-contained search query using the chat history. "
             "Resolve all pronouns and references (e.g. 'it', 'they', 'that professor'). "
             "Return ONLY the rewritten query."
             ),
            ("placeholder", "{history}"),
            ("human", "{question}"),
        ])

        chat_model = self.chat_model  # local ref for lambda capture

        rewrite_chain = (
            rewrite_prompt
            | chat_model
            | RunnableLambda(lambda m: m.content.strip())
        )

        rag_chain = (
            {
                "docs":     itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "history":  itemgetter("history"),
            }
            | RunnableLambda(lambda x: {
                **x,
                "context": "\n\n---\n\n".join(d.page_content for d in x["docs"]),
            })
            | RunnableLambda(lambda x: {
                "answer": chat_model.invoke(
                    rag_prompt.invoke({
                        "context":  x["context"],
                        "question": x["question"],
                        "history":  x["history"],
                    })
                ).content,
                "sources": x["docs"],
            })
        )

        rag_chain_with_rewrite = (
            RunnablePassthrough()
            | RunnablePassthrough.assign(question=rewrite_chain)
            | rag_chain
        )

        self._store: dict = {}

        self.conversational_rag = RunnableWithMessageHistory(
            rag_chain_with_rewrite,
            self._get_history,
            input_messages_key="question",
            output_messages_key="answer",
            history_messages_key="history",
        )

    def _get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

    def chat(self, message: str, session_id: str = "diem-session") -> str:
        result = self.conversational_rag.invoke(
            {"question": message},
            config={"configurable": {"session_id": session_id}},
        )
        answer  = result["answer"]
        sources = result.get("sources", [])
        if sources:
            seen, unique = set(), []
            for doc in sources:
                url = doc.metadata.get("source", "")
                if url and url not in seen:
                    seen.add(url)
                    unique.append(url)
            if unique:
                answer += "\n\n**Sources:**\n" + "\n".join(f"- {u}" for u in unique)
        return answer
