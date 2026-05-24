"""
LangSmith Studio / langgraph dev entrypoint.

Exposes `diem_rag_graph` as a module-level variable pointing to a compiled
StateGraph without a custom checkpointer (the platform handles persistence).

Kept separate from brain.py so DiemBrain has no langgraph-dev-specific code.
"""

from src.agent.brain import DiemBrain
from src.rag_hybrid import QdrantRAG
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from config import QDRANT_HOST, QDRANT_PORT, OPENROUTER_API_KEY

    # Initialize the Hybrid RAG system
    _hybrid_rag = QdrantRAG(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        openrouter_api_key=OPENROUTER_API_KEY
    )

    _brain = DiemBrain(_hybrid_rag)
    # No checkpointer: langgraph dev platform handles persistence automatically
    diem_rag_graph = _brain._build_graph(_brain._tools, checkpointer=None)
    logger.info("graph_dev: diem_rag_graph (Hybrid) loaded successfully")
except Exception as _e:
    logger.warning(f"graph_dev: could not build graph ({_e})")
    diem_rag_graph = None
