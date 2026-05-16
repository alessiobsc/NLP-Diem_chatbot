"""
LangSmith Studio / langgraph dev entrypoint.

Exposes `diem_rag_graph` as a module-level variable pointing to a compiled
StateGraph without a custom checkpointer (the platform handles persistence).

Kept separate from brain.py so DiemBrain has no langgraph-dev-specific code.
"""

from langchain_chroma import Chroma

from app import embedding_model
from src.agent.brain import DiemBrain
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from config import CHROMA_DIR_NAME, COLLECTION_NAME

    _vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR_NAME,
    )
    _brain = DiemBrain(_vectorstore)
    # No checkpointer: langgraph dev platform handles persistence automatically
    diem_rag_graph = _brain._build_graph(_brain._tools, checkpointer=None)
    logger.info("graph_dev: diem_rag_graph loaded successfully")
except Exception as _e:
    logger.warning(f"graph_dev: could not build graph ({_e})")
    diem_rag_graph = None
