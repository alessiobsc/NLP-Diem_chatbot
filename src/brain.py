"""
Core AI Brain module for the DIEM Chatbot.
Module-level symbols (embedding_model, reranker, rerank, _format_context) are kept unchanged
so ingestion scripts and evaluation/tester.py continue to import without modification.
"""

import uuid
from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers.multi_vector import SearchType
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver

from sentence_transformers import CrossEncoder

from config import (
    PARENT_STORE_DIR,
    EMBEDDING_MODEL_NAME,
    CROSS_ENCODER_MODEL_NAME,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    BI_ENCODER_K,
    CROSS_ENCODER_K,
    RETRIEVER_SCORE_THRESHOLD,
    DEFAULT_SESSION_ID,
)
from src.models import _build_agent_model, _build_chat_model
from src.tools import build_tools
from src.middleware import build_middleware
from src.prompts import SYSTEM_PROMPT, REJECTION_TAGS
from src.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level symbols: unchanged (imported by ingestion scripts and tester)
# ─────────────────────────────────────────────────────────────────────────────

class E5HuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """E5 model wrapper: prepends 'query:' / 'passage:' prefixes."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"query: {text}")


logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = E5HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},
)

logger.info(f"Initializing reranker model: {CROSS_ENCODER_MODEL_NAME}")
reranker = CrossEncoder(CROSS_ENCODER_MODEL_NAME)


def rerank(query: str, documents: List[Document], top_n: int = CROSS_ENCODER_K) -> List[Document]:
    if not documents:
        return []
    pairs = [[query, d.page_content] for d in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    out = []
    for i, (d, s) in enumerate(ranked[:top_n]):
        d.metadata["relevance_score"] = float(s)
        out.append(d)
    logger.info(f"Reranked: top {len(out)} of {len(documents)}")
    return out


def _format_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
    docs: List[Document] = inputs.get("docs", [])
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        blocks.append(
            f"<document>\n<source>{source}</source>\n"
            f"<content>\n{doc.page_content}\n</content>\n</document>"
        )
    return {**inputs, "context": "\n\n".join(blocks)}


# ─────────────────────────────────────────────────────────────────────────────
# DiemBrain
# ─────────────────────────────────────────────────────────────────────────────

class DiemBrain:
    """Agentic RAG system for DIEM. Uses create_agent with 4 tools and middleware."""

    def __init__(self, vectorstore: Chroma) -> None:
        self._last_docs: List[Document] = []
        # Per-session history: only HumanMessage + final AIMessage (no ToolMessages).
        # MemorySaver would accumulate ToolMessages causing the agent to skip retrieve()
        # on subsequent turns, so history is managed manually here.
        self._history: Dict[str, List[BaseMessage]] = {}

        self._agent_model = _build_agent_model()
        self._generation_model = _build_chat_model()
        self._retriever = self._build_retriever(vectorstore)

        tools = build_tools(
            self._retriever,
            self._generation_model,
            self,
            None,  # rag_prompt unused: answer tool builds its own prompt inline
        )
        middleware = build_middleware(self._generation_model)

        self._agent = create_agent(
            self._agent_model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT + (
                "\n6. TOOL USAGE: Call retrieve() whenever you need information from the "
                "DIEM knowledge base to answer the question. For any new factual question "
                "about DIEM, call retrieve() first. If the question can be answered from "
                "the conversation history (e.g. asking to repeat or clarify a previous answer), "
                "you may answer directly without retrieve(). "
                "After retrieve(), call answer() with the retrieved context and return its "
                "output verbatim as your final response."
            ),
            middleware=middleware,
            checkpointer=MemorySaver(),  # stateless per-turn (unique thread_id each call)
        )
        logger.info("DiemBrain (agentic) initialization complete")

    # ── retriever ────────────────────────────────────────────────────────────

    def _build_retriever(self, vectorstore: Chroma) -> ParentDocumentRetriever:
        parent_doc_store = create_kv_docstore(LocalFileStore(str(PARENT_STORE_DIR)))
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
        )
        return ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=parent_doc_store,
            child_splitter=child_splitter,
            search_type=SearchType.similarity_score_threshold,
            search_kwargs={"k": BI_ENCODER_K, "score_threshold": RETRIEVER_SCORE_THRESHOLD},
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _format_sources(self, sources: List[Document]) -> str:
        if not sources:
            return ""
        seen, urls = set(), []
        for doc in sources:
            url = doc.metadata.get("source", "")
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
        if urls:
            return "\n\n**Sources:**\n" + "\n".join(f"- {u}" for u in urls)
        return ""

    def _invoke_config(self) -> dict:
        # Unique thread_id per call: MemorySaver is stateless between turns.
        # History is injected manually via _history to exclude ToolMessages.
        return {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 15}

    def _strip_rejection_tags(self, text: str) -> str:
        for tag in REJECTION_TAGS:
            if text.startswith(tag):
                return text[len(tag):].lstrip()
        return text

    def _input_messages(self, session_id: str, message: str) -> List[BaseMessage]:
        return self._history.get(session_id, []) + [HumanMessage(message)]

    def _save_turn(self, session_id: str, message: str, answer: str) -> None:
        if session_id not in self._history:
            self._history[session_id] = []
        self._history[session_id].append(HumanMessage(message))
        self._history[session_id].append(AIMessage(answer))

    # ── public API ────────────────────────────────────────────────────────────

    def chat(self, message: str, session_id: str = DEFAULT_SESSION_ID) -> str:
        """Non-streaming chat. Returns answer string (with appended sources)."""
        logger.info(f"chat | session={session_id} | msg={message[:60]}")
        self._last_docs = []
        try:
            result = self._agent.invoke(
                {"messages": self._input_messages(session_id, message)},
                config=self._invoke_config(),
            )
            raw = result["messages"][-1].content
            answer = self._strip_rejection_tags(raw)
            is_rejection = raw != answer
            self._save_turn(session_id, message, answer)
            if is_rejection:
                return answer
            return answer + self._format_sources(self._last_docs)
        except Exception as e:
            logger.exception(f"chat error: {e}")
            return "Mi dispiace, si è verificato un errore."

    def chat_stream(self, message: str, session_id: str = DEFAULT_SESSION_ID):
        """Streaming generator: yields LLM tokens then appends sources.
        Resets answer accumulator on each tools node to discard tool-call
        decision text that leaks as content before tool_call_chunks appear.
        """
        logger.info(f"chat_stream | session={session_id}")
        self._last_docs = []
        answer = ""
        try:
            for chunk, metadata in self._agent.stream(
                {"messages": self._input_messages(session_id, message)},
                config=self._invoke_config(),
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node", "")
                if node == "tools":
                    answer = ""  # discard pre-tool agent content (<tool_call> leakage)
                    continue
                is_tool_call = bool(getattr(chunk, "tool_call_chunks", None))
                if node in ("model", "agent") and not is_tool_call and hasattr(chunk, "content") and chunk.content:
                    answer += chunk.content
                    yield answer
        except Exception as e:
            logger.error(f"chat_stream error: {e}")
            yield "Mi dispiace, si è verificato un errore."
            return

        stripped = self._strip_rejection_tags(answer)
        is_rejection = stripped != answer
        self._save_turn(session_id, message, stripped if is_rejection else answer)
        if is_rejection:
            if stripped:
                yield stripped
            return
        sources_md = self._format_sources(self._last_docs)
        if sources_md and answer:
            yield answer + sources_md

    def chat_eval(self, message: str, session_id: str = DEFAULT_SESSION_ID) -> Dict[str, Any]:
        """Evaluation interface: returns {'answer': str, 'sources': List[Document]}."""
        logger.info(f"chat_eval | session={session_id}")
        self._last_docs = []
        try:
            result = self._agent.invoke(
                {"messages": self._input_messages(session_id, message)},
                config=self._invoke_config(),
            )
            raw = result["messages"][-1].content
            answer = self._strip_rejection_tags(raw)
            self._save_turn(session_id, message, answer)
            return {"answer": answer, "sources": list(self._last_docs)}
        except Exception as e:
            logger.exception(f"chat_eval error: {e}")
            return {"answer": "", "sources": [], "error": f"{type(e).__name__}: {e}"}

    def get_history(self, session_id: str) -> List[BaseMessage]:
        """Return clean message history (HumanMessage + AIMessage only) for session."""
        return list(self._history.get(session_id, []))
