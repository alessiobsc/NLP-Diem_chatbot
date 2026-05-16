"""
Core AI Brain module for the DIEM Chatbot.

Module-level symbols (embedding_model, reranker, rerank, _format_context) are kept
so ingestion scripts and tester.py continue to import without modification.
"""

import uuid
from typing import Any, Annotated, Dict, List

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers.multi_vector import SearchType
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from sentence_transformers import CrossEncoder
from typing_extensions import TypedDict

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
    MAX_TOOL_CALLS,
)
from src.models import _build_agent_model, _build_chat_model
from src.tools import build_tools
from src.middleware import (
    ScopeGuardrail, OffensiveContentGuardrail, redact_pii, _SCOPE_REJECTION,
)
from src.prompts import SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT, REJECTION_TAGS
from src.logger import get_logger

logger = get_logger(__name__)


# ── Module-level symbols: unchanged (imported by ingestion scripts and tester) ──

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
    for d, s in ranked[:top_n]:
        d.metadata["relevance_score"] = float(s)
        out.append(d)
    logger.info(f"Reranked: top {len(out)} of {len(documents)}")
    return out


def _format_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats retrieved documents into a single context string.

    Args:
        inputs (Dict[str, Any]): Dictionary containing 'docs'.

    Returns:
        Dict[str, Any]: Inputs augmented with the 'context' string.
    """
    docs: List[Document] = inputs.get("docs", [])
    logger.debug(f"Formatting context from {len(docs)} reranked documents")

    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        content = _strip_context_header_from_content(doc)
        block = (
            "<document>\n"
            f"<source>{source}</source>\n"
            f"<content>\n{content}\n</content>\n"
            "</document>"
        )
        formatted_docs.append(block)

    context = "\n\n".join(formatted_docs)
    if docs:
        logger.debug(f"Total formatted context length: {len(context)} characters")
    return {**inputs, "context": context}


def _strip_context_header_from_content(doc: Document) -> str:
    """
    Remove generated retrieval headers before sending evidence to the answer model.
    """
    content = doc.page_content or ""
    header = doc.metadata.get("context_header", "")

    if isinstance(header, str) and header:
        stripped = content.lstrip()
        if stripped.startswith(header):
            return stripped[len(header):].lstrip()

    stripped = content.lstrip()
    if stripped.lower().startswith("context:"):
        lines = stripped.splitlines()
        if lines:
            return "\n".join(lines[1:]).lstrip()

    return content



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


# ── Graph State ───────────────────────────────────────────────────────────────

class DiemState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_call_count: int      # increments each time the tools node fires
    retrieved_context: str    # latest retrieve output, passed to generate node
    last_docs: List[Document] # latest retrieved docs, used for source URL formatting


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Extract plain text from a message content that may be a string or a list of content blocks.

    LangSmith Studio sends HumanMessage.content as a list of dicts
    (e.g. [{"type": "text", "text": "..."}]) instead of a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


# ── Routing functions (module-level so tests can import them directly) ────────

def _route_scope(state: DiemState) -> str:
    """After scope_guard: an injected AIMessage means rejection → END."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        return "__end__"
    return "retrieve_node"


def _route_agent(state: DiemState) -> str:
    """After agent: go to tools if tool_calls present and under safety cap, else generate."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None) and state["tool_call_count"] < MAX_TOOL_CALLS:
        return "tools"
    return "generate"


# ── DiemBrain ─────────────────────────────────────────────────────────────────

class DiemBrain:
    """Agentic RAG system for DIEM using an explicit LangGraph StateGraph."""

    def __init__(self, vectorstore: Chroma) -> None:
        self._last_docs: List[Document] = []

        self._agent_model = _build_agent_model()
        self._generation_model = _build_chat_model()
        self._retriever = self._build_retriever(vectorstore)

        self._tools = build_tools(self._retriever, self._generation_model, self)

        # Bind tools to 32b so it can emit tool_calls in its AIMessage output
        self._agent_model_with_tools = self._agent_model.bind_tools(self._tools)

        self._scope_guardrail = ScopeGuardrail(self._generation_model)
        self._offensive_guardrail = OffensiveContentGuardrail(self._generation_model)

        # Build generation prompt once (reused every _node_generate call)
        self._generate_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "<context>\n{context}\n</context>\n\n<instruction>\n{question}\n</instruction>"),
        ])

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

        g.add_node("scope_guard", self._node_scope_guard)
        g.add_node("retrieve_node", self._node_retrieve)
        g.add_node("agent", self._node_agent)
        g.add_node("tools", self._make_tools_node(tools))
        g.add_node("generate", self._node_generate)
        g.add_node("output_guard", self._node_output_guard)

        g.set_entry_point("scope_guard")
        g.add_conditional_edges("scope_guard", _route_scope)
        g.add_edge("retrieve_node", "agent")
        g.add_conditional_edges("agent", _route_agent)
        g.add_edge("tools", "agent")
        g.add_edge("generate", "output_guard")
        g.add_edge("output_guard", END)

        compile_kwargs: dict = {"name": "diem_rag_graph"}
        if checkpointer is not None:
            compile_kwargs["checkpointer"] = checkpointer
        return g.compile(**compile_kwargs)

    # ── Node implementations ──────────────────────────────────────────────────

    def _node_scope_guard(self, state: DiemState) -> dict:
        """Reject out-of-scope queries before retrieval.

        Returns {} (no state change) if in scope → _route_scope sends to retrieve_node.
        Returns {messages: [AIMessage(rejection)]} if OOT → _route_scope sends to END.
        """
        question = next(
            (_extract_text(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        if not self._scope_guardrail.check(question):
            return {"messages": [AIMessage(content=_SCOPE_REJECTION)]}
        return {}

    def _node_retrieve(self, state: DiemState) -> dict:
        """Mandatory retrieval step — always runs before the 32b agent loop.

        Injects a fake AIMessage (retrieve tool call) + ToolMessage (context) into
        `messages` so the 32b agent sees context in the standard tool-call format
        it was trained on. A bare ToolMessage without a preceding AIMessage tool_call
        would violate the OpenAI message schema and cause API errors.
        """
        query = next(
            (_extract_text(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        docs = self._retriever.invoke(query)
        reranked = rerank(query, docs) if docs else []
        context = _format_context({"docs": reranked, "question": query, "history": []})["context"]

        # Sync to instance attr so chat_stream can read docs after streaming completes
        self._last_docs = reranked

        # Simulate the retrieve call so the 32b sees a proper tool call + result pair
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
        """32b decides whether to call more tools or let generate handle the answer.

        SYSTEM_PROMPT injected as SystemMessage at position 0, which is valid for
        all providers. The retrieved context is already in messages from retrieve_node.
        """
        # AGENT_SYSTEM_PROMPT: routing only — no response generation instructions
        system = SystemMessage(content=AGENT_SYSTEM_PROMPT)
        response = self._agent_model_with_tools.invoke([system] + list(state["messages"]))
        return {"messages": [response]}

    def _node_generate(self, state: DiemState) -> dict:
        """9b generates the final answer from retrieved context + SYSTEM_PROMPT.

        Uses state['retrieved_context'] directly (always the latest retrieve output)
        rather than searching through messages, for reliability.
        """
        context = state.get("retrieved_context", "")
        question = next(
            (_extract_text(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        logger.info(f"generate node: context_len={len(context)} question_len={len(question)}")
        result = self._generation_model.invoke(
            self._generate_prompt.invoke({"context": context, "question": question})
        )
        return {"messages": [AIMessage(content=result.content)]}

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

        # Offensive check (keyword fast path inside .check(), LLM only on hit)
        replacement = self._offensive_guardrail.check(content)
        if replacement is not None:
            return {"messages": [AIMessage(id=last_ai.id, content=replacement)]}

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
        only the 9b final answer reaches the user. Source URLs yielded as final chunk.
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
                # Capture scope rejection text (graph ends before generate runs)
                if node == "scope_guard" and isinstance(chunk, AIMessage) and chunk.content:
                    rejection = chunk.content
                # Only stream tokens produced by the generate node (9b final answer)
                if node == "generate" and hasattr(chunk, "content") and chunk.content:
                    if not getattr(chunk, "tool_call_chunks", None):
                        content = chunk.content
                        # Strip rejection tags inline so they never reach the user
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
