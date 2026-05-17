"""
Core AI Brain module for the DIEM Chatbot.

Module-level symbols (rerank, format_context) are kept so ingestion scripts
and tester.py continue to import without modification.
"""
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers.multi_vector import SearchType
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from config import (
    PARENT_STORE_DIR,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    BI_ENCODER_K,
    RETRIEVER_SCORE_THRESHOLD,
    DEFAULT_SESSION_ID,
    MAX_RETRIEVE_CALLS,
)
from src.agent.state import DiemState
from src.agent.nodes import DiemNodes
from src.agent.utils import extract_text, format_context, rewrite_query
from src.agent.init_models import build_agent_model, build_lightweight_model
from src.agent.tools import build_tools
from src.encoders.reranker import rerank
from src.middleware import ScopeGuardrail, OffensiveContentGuardrail
from src.prompts import AGENT_SYSTEM_PROMPT, REJECTION_TAGS
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Routing functions (module-level so tests can import them directly) ────────

def _route_input(state: DiemState) -> str:
    """After input_guard: an injected AIMessage means offensive input → END."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        return "__end__"
    return "scope_guard"


def _route_scope(state: DiemState) -> str:
    """After scope_guard: an injected AIMessage means rejection → END."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        return "__end__"
    return "reset_state"


def _route_agent(state: DiemState) -> str:
    """After agent: tools, force_answer (retrieve cap), forced_retrieve, or output_guard."""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if not tool_calls:
        content = (last.content if isinstance(last.content, str) else "").strip()
        # Agent produced nothing without ever calling retrieve → force a retrieve
        if not content and state["tool_call_count"] == 0 and not state["retrieved_context"]:
            return "forced_retrieve"
        return "output_guard"
    # Cap only applies when agent wants to call retrieve again
    wants_retrieve = any(tc["name"] == "retrieve" for tc in tool_calls)
    if wants_retrieve and state["tool_call_count"] >= MAX_RETRIEVE_CALLS:
        return "force_answer"
    return "tools"


# ── DiemBrain ─────────────────────────────────────────────────────────────────

class DiemBrain(DiemNodes):
    """Agentic RAG system for DIEM using an explicit LangGraph StateGraph."""

    def __init__(self, vectorstore: Chroma) -> None:
        self._last_docs: List[Document] = []

        self._agent_model = build_agent_model()
        self._lightweight_model = build_lightweight_model()
        self._retriever = self._build_retriever(vectorstore)

        self._tools = build_tools(self._retriever, self._lightweight_model, self)

        # Bind tools to agent model so it can emit tool_calls in its AIMessage output
        self._agent_model_with_tools = self._agent_model.bind_tools(self._tools)

        self._scope_guardrail = ScopeGuardrail(self._lightweight_model)
        self._offensive_guardrail = OffensiveContentGuardrail()

        self._graph = self._build_graph(self._tools, checkpointer=MemorySaver())
        logger.info("DiemBrain (explicit StateGraph) initialization complete")

    # ── Retriever ─────────────────────────────────────────────────────────────

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

    # ── Graph construction ────────────────────────────────────────────────────

    def _make_tools_node(self, tools: list):
        """Wrap ToolNode to also update tool_call_count and retrieved_context in state.

        Standard ToolNode only appends ToolMessages to `messages`. This wrapper
        additionally increments the retrieve counter and syncs retrieved_context
        and last_docs into state when retrieve() was called.
        """
        tool_node = ToolNode(tools)

        def tools_node(state: DiemState) -> dict:
            result = tool_node.invoke(state)

            # Only retrieve calls count toward the cap; rewrite/summarize/calculate are free
            last_ai = state["messages"][-1]
            tool_calls = getattr(last_ai, "tool_calls", [])
            retrieve_call = next(
                (tc for tc in tool_calls if tc["name"] == "retrieve"), None
            )
            new_retrieve_count = state["tool_call_count"] + (1 if retrieve_call else 0)
            updates = {"tool_call_count": new_retrieve_count}

            called_names = [tc["name"] for tc in tool_calls]
            logger.info(
                f"tools_node | called={called_names} | retrieve_count={new_retrieve_count}"
            )

            if retrieve_call:
                retrieve_msg = next(
                    (m for m in result["messages"]
                     if isinstance(m, ToolMessage) and m.tool_call_id == retrieve_call["id"]),
                    None,
                )
                if retrieve_msg:
                    updates["retrieved_context"] = retrieve_msg.content
                    # self._last_docs was just updated by the retrieve tool's closure
                    updates["last_docs"] = list(self._last_docs)

            return {**result, **updates}

        return tools_node

    def _build_graph(self, tools: list, checkpointer=None):
        """Build and compile the explicit StateGraph.

        checkpointer=MemorySaver() for normal app use; None for langgraph dev
        (platform handles persistence when no checkpointer is provided).
        name='diem_rag_graph' makes every node visible by name in LangSmith Studio.
        """
        g = StateGraph(DiemState)

        g.add_node("input_guard", self._node_input_guard)
        g.add_node("scope_guard", self._node_scope_guard)
        g.add_node("reset_state", self._node_reset_state)
        g.add_node("agent", self._node_agent)
        g.add_node("tools", self._make_tools_node(tools))
        g.add_node("forced_retrieve", self._node_forced_retrieve)
        g.add_node("force_answer", self._node_force_answer)
        g.add_node("output_guard", self._node_output_guard)

        g.set_entry_point("input_guard")
        g.add_conditional_edges("input_guard", _route_input)
        g.add_conditional_edges("scope_guard", _route_scope)
        g.add_edge("reset_state", "agent")
        g.add_conditional_edges("agent", _route_agent)
        g.add_edge("tools", "agent")
        g.add_edge("forced_retrieve", "agent")
        g.add_edge("force_answer", "output_guard")
        g.add_edge("output_guard", END)

        compile_kwargs: dict = {"name": "diem_rag_graph"}
        if checkpointer is not None:
            compile_kwargs["checkpointer"] = checkpointer
        return g.compile(**compile_kwargs)

    # ── Helpers ───────────────────────────────────────────────────────────────

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

    def _invoke_config(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}, "recursion_limit": 20}

    def _strip_rejection_tags(self, text: str) -> str:
        for tag in REJECTION_TAGS:
            if text.startswith(tag):
                return text[len(tag):].lstrip()
        return text

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, message: str, session_id: str = DEFAULT_SESSION_ID) -> str:
        """Non-streaming chat. Returns answer string with appended source URLs."""
        logger.info(f"chat | session={session_id} | msg={message[:60]}")
        try:
            result = self._graph.invoke(
                {"messages": [HumanMessage(message)]},
                config=self._invoke_config(session_id),
            )
            self._last_docs = list(result.get("last_docs", []))
            raw = result["messages"][-1].content
            answer = self._strip_rejection_tags(raw)
            is_rejection = raw != answer
            if is_rejection:
                return answer
            return answer + self._format_sources(self._last_docs)
        except Exception as e:
            logger.exception(f"chat error: {e}")
            return "Mi dispiace, si è verificato un errore."

    def chat_stream(self, message: str, session_id: str = DEFAULT_SESSION_ID):
        """Streaming generator. Yields tokens from the agent node only.

        Tokens from guard/reset/tools nodes are discarded — only the final answer
        reaches the user. Source URLs yielded as final chunk.
        For scope rejections the rejection text is yielded at end.
        """
        logger.info(f"chat_stream | session={session_id}")
        answer = ""
        rejection = ""
        try:
            for chunk, metadata in self._graph.stream(
                {"messages": [HumanMessage(message)]},
                config=self._invoke_config(session_id),
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node", "")
                if node in ("input_guard", "scope_guard") and isinstance(chunk, AIMessage) and chunk.content:
                    rejection = chunk.content
                if node == "agent" and hasattr(chunk, "content") and chunk.content:
                    if not getattr(chunk, "tool_call_chunks", None):
                        is_partial = isinstance(chunk, AIMessageChunk)
                        # stream_mode="messages" emits both per-token AIMessageChunks and the
                        # final complete AIMessage written to state — skip the latter to avoid duplication.
                        if is_partial or not answer:
                            content = chunk.content
                            for tag in REJECTION_TAGS:
                                content = content.replace(tag, "")
                            answer += content
                            if content:
                                yield content
        except Exception as e:
            logger.error(f"chat_stream error: {e}")
            yield "Mi dispiace, si è verificato un errore."
            return

        if not answer:
            if rejection:
                yield self._strip_rejection_tags(rejection)
            return

        sources_md = self._format_sources(self._last_docs)
        if sources_md:
            yield sources_md

    def chat_eval(self, message: str, session_id: str = DEFAULT_SESSION_ID) -> Dict[str, Any]:
        """Evaluation interface. Returns {'answer': str, 'sources': List[Document]}."""
        logger.info(f"chat_eval | session={session_id}")
        try:
            result = self._graph.invoke(
                {"messages": [HumanMessage(message)]},
                config=self._invoke_config(session_id),
            )
            raw = result["messages"][-1].content
            answer = self._strip_rejection_tags(raw)
            return {"answer": answer, "sources": list(result.get("last_docs", []))}
        except Exception as e:
            logger.exception(f"chat_eval error: {e}")
            return {"answer": "", "sources": [], "error": f"{type(e).__name__}: {e}"}

    def get_history(self, session_id: str) -> List[BaseMessage]:
        """Return full message history from MemorySaver for this session."""
        try:
            state = self._graph.get_state({"configurable": {"thread_id": session_id}})
            return list(state.values.get("messages", []))
        except Exception:
            return []
