"""
Core AI Brain module for the DIEM Chatbot.

Module-level symbols (embedding_model, reranker, rerank, _format_context) are kept
so ingestion scripts and tester.py continue to import without modification.
"""
import uuid
from typing import Any, Annotated, Dict, List, TypedDict
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers.multi_vector import SearchType
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, AIMessageChunk, SystemMessage, ToolMessage,
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
    DEFAULT_SESSION_ID, MAX_TOOL_CALLS
)
from src.embeddings.reranker import rerank
from src.agent.utils import extract_text, format_context, rewrite_query
from src.agent.init_models import build_agent_model, build_lightweight_model
from src.agent.tools import build_tools
from src.middleware import (
    ScopeGuardrail, OffensiveContentGuardrail, redact_pii, _SCOPE_REJECTION,
)
from src.prompts import AGENT_SYSTEM_PROMPT, REJECTION_TAGS
from src.utils.logger import get_logger
from langgraph.graph.message import add_messages

logger = get_logger(__name__)

# ── Module-level symbols: unchanged (imported by ingestion scripts and tester) ──

# ── Graph State ───────────────────────────────────────────────────────────────

class DiemState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_call_count: int      # increments each time the tools node fires
    retrieved_context: str    # latest retrieve output, passed to generate node
    last_docs: List[Document] # latest retrieved docs, used for source URL formatting


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
    return "retrieve_node"


def _route_agent(state: DiemState) -> str:
    """After agent: tools if tool_calls present, else output_guard."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None) and state["tool_call_count"] < MAX_TOOL_CALLS:
        return "tools"
    return "output_guard"


# ── DiemBrain ─────────────────────────────────────────────────────────────────

class DiemBrain:
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
        additionally increments the safety counter and, when `retrieve` was called,
        syncs the fresh context into state so generate node always uses the latest.
        """
        tool_node = ToolNode(tools)

        def tools_node(state: DiemState) -> dict:
            result = tool_node.invoke(state)

            updates = {"tool_call_count": state["tool_call_count"] + 1}

            # Check whether retrieve was called in the last AIMessage's tool_calls
            last_ai = state["messages"][-1]
            tool_calls = getattr(last_ai, "tool_calls", [])
            retrieve_call = next(
                (tc for tc in tool_calls if tc["name"] == "retrieve"), None
            )

            if retrieve_call:
                # Match ToolMessage by tool_call_id (handles batch tool calls correctly)
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
        g.add_node("retrieve_node", self._node_retrieve)
        g.add_node("agent", self._node_agent)
        g.add_node("tools", self._make_tools_node(tools))
        g.add_node("output_guard", self._node_output_guard)

        g.set_entry_point("input_guard")
        g.add_conditional_edges("input_guard", _route_input)
        g.add_conditional_edges("scope_guard", _route_scope)
        g.add_edge("retrieve_node", "agent")
        g.add_conditional_edges("agent", _route_agent)
        g.add_edge("tools", "agent")
        g.add_edge("output_guard", END)

        compile_kwargs: dict = {"name": "diem_rag_graph"}
        if checkpointer is not None:
            compile_kwargs["checkpointer"] = checkpointer
        return g.compile(**compile_kwargs)

    # ── Node implementations ──────────────────────────────────────────────────

    def _block_if_offensive(self, content: str, msg_id: str | None = None) -> dict:
        """Return state update that replaces/injects AIMessage if content is offensive, else {}."""
        replacement = self._offensive_guardrail.check(content)
        if replacement is None:
            return {}
        kwargs: dict = {"content": replacement}
        if msg_id is not None:
            kwargs["id"] = msg_id
        return {"messages": [AIMessage(**kwargs)]}

    def _node_input_guard(self, state: DiemState) -> dict:
        """Reject offensive user input before scope check."""
        question = next(
            (extract_text(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        return self._block_if_offensive(question)

    def _node_scope_guard(self, state: DiemState) -> dict:
        """Reject out-of-scope queries before retrieval.

        Returns {} (no state change) if in scope → _route_scope sends to retrieve_node.
        Returns {messages: [AIMessage(rejection)]} if OOT → _route_scope sends to END.
        """
        question = next(
            (extract_text(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        if not self._scope_guardrail.check(question):
            return {"messages": [AIMessage(content=_SCOPE_REJECTION)]}
        return {}

    def _node_retrieve(self, state: DiemState) -> dict:
        """Mandatory retrieval step — always runs before the agent loop.

        Injects a fake AIMessage (retrieve tool call) + ToolMessage (context) into
        `messages` so the agent sees context in the standard tool-call format
        it was trained on. A bare ToolMessage without a preceding AIMessage tool_call
        would violate the OpenAI message schema and cause API errors.
        """
        raw_query = next(
            (extract_text(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        has_history = any(
            isinstance(m, HumanMessage)
            for m in state["messages"][:-1]
            if not getattr(m, "tool_calls", None)
        )
        query = rewrite_query(raw_query, state, self._lightweight_model) if has_history else raw_query
        docs = self._retriever.invoke(query)
        reranked = rerank(query, docs) if docs else []
        context = format_context({"docs": reranked, "question": query, "history": []})["context"]

        # Sync to instance attr so chat_stream can read docs after streaming completes
        self._last_docs = reranked

        # Simulate the retrieve call so the agent sees a proper tool call + result pair
        tool_call_id = str(uuid.uuid4())
        fake_ai = AIMessage(
            content="",
            tool_calls=[{
                "name": "retrieve",
                "args": {"query": query},
                "id": tool_call_id,
                "type": "tool_call",
            }],
        )
        tool_msg = ToolMessage(
            content=context,
            tool_call_id=tool_call_id,
            name="retrieve",
        )

        return {
            "messages": [fake_ai, tool_msg],
            "retrieved_context": context,
            "last_docs": reranked,
            "tool_call_count": 0,  # reset at the start of each new turn
        }

    def _node_agent(self, state: DiemState) -> dict:
        """Agent decides whether to call more tools or let generate handle the answer.

        SYSTEM_PROMPT injected as SystemMessage at position 0, which is valid for
        all providers. The retrieved context is already in messages from retrieve_node.
        """
        # AGENT_SYSTEM_PROMPT: routing only — no response generation instructions
        system = SystemMessage(content=AGENT_SYSTEM_PROMPT)
        response = self._agent_model_with_tools.invoke([system] + list(state["messages"]))
        return {"messages": [response]}

    def _node_output_guard(self, state: DiemState) -> dict:
        """Offensive content check + PII redaction on the final AIMessage.

        Returns {} if content is clean (no state change).
        Returns updated messages if content was replaced or redacted.
        """
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        if last_ai is None:
            return {}

        content = last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content)

        blocked = self._block_if_offensive(content, msg_id=last_ai.id)
        if blocked:
            return blocked

        # PII: email redact / credit card block via regex (no LLM call)
        redacted = redact_pii(content)
        if redacted != content:
            return {"messages": [AIMessage(id=last_ai.id, content=redacted)]}

        return {}

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
            # Sync last_docs from final graph state
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
        """Streaming generator. Yields tokens from the generate node only.

        Tokens from scope_guard / retrieve_node / agent / tools nodes are discarded —
        only the generator's final answer reaches the user. Source URLs yielded as final chunk.
        For scope rejections (no generate node), the rejection text is yielded at end.
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
                # Capture early rejection text (input_guard or scope_guard)
                if node in ("input_guard", "scope_guard") and isinstance(chunk, AIMessage) and chunk.content:
                    rejection = chunk.content
                # Stream tokens from agent node only — tool_call_chunks guard filters
                # out intermediate tool-call emissions from the agent routing steps.
                if node == "agent" and hasattr(chunk, "content") and chunk.content:
                    if not getattr(chunk, "tool_call_chunks", None):
                        is_partial = isinstance(chunk, AIMessageChunk)
                        # LangGraph stream_mode="messages" emits both per-token
                        # AIMessageChunk objects and the final complete AIMessage
                        # written to state — skip the latter to avoid duplication.
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

        # If generate node never ran (scope rejection), yield the rejection text
        if not answer:
            if rejection:
                yield self._strip_rejection_tags(rejection)
            return

        # self._last_docs updated by _node_retrieve (or _make_tools_node on re-retrieve)
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
