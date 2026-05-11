"""
Centralized Configuration Module for the DIEM Chatbot system.

This module uses python-dotenv to load environment variables and provides a single
source of truth for all configurations (paths, model settings, RAG parameters, etc.).
It follows the Singleton pattern via module-level constants to ensure consistency
across the entire application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Load environment variables from .env file if it exists
load_dotenv()

# =============================================================================
# DIRECTORY & PATH CONFIGURATION
# =============================================================================
# Define the root of the project to build absolute paths safely
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ChromaDB Storage paths
CHROMA_DIR_NAME: str = "chroma_diem"
CHROMA_DIR: Path = PROJECT_ROOT / CHROMA_DIR_NAME
PARENT_STORE_DIR: Path = CHROMA_DIR / "parent_store"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# LLM Settings (Ollama)
OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Embedding Model Settings (HuggingFace)
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")

# Cross-Encoder Model Settings
CROSS_ENCODER_MODEL_NAME: str = os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# =============================================================================
# RAG & VECTOR DATABASE CONFIGURATION
# =============================================================================
COLLECTION_NAME: str = "diem_collect_HeaderContext_new_Italiano"
DEFAULT_SESSION_ID: str = "diem-session"

# Retrieval Settings
# BI_ENCODER_K is the number of documents retrieved in the first stage (fast retrieval)
BI_ENCODER_K: int = int(os.getenv("BI_ENCODER_K", "20"))
# CROSS_ENCODER_K is the number of documents kept after reranking in the second stage (precision reranking)
CROSS_ENCODER_K: int = int(os.getenv("CROSS_ENCODER_K", "3"))

RETRIEVER_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.5"))

# Document Splitting Settings (Parent-Child Strategy)
# Parent Document Settings (Broad context)
PARENT_CHUNK_SIZE: int = 2000
PARENT_CHUNK_OVERLAP: int = 200

# Child Document Settings (Precise retrieval)
CHILD_CHUNK_SIZE: int = 400
CHILD_CHUNK_OVERLAP: int = 50

# Ingestion Batching
MAX_CHILD_CHUNKS_PER_BATCH: int = 4000
