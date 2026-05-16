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
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")
LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_FILE: Path = LOG_DIR / "chatbot.log"

# Log rotation settings to prevent infinite growth
MAX_LOG_SIZE_MB: int = int(os.getenv("MAX_LOG_SIZE_MB", "10"))
LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))


# =============================================================================
# PROVIDER SELECTION
# =============================================================================
# LLM provider: "local" (default) or "openrouter"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "local")

if LLM_PROVIDER not in ["local", "openrouter"]:
    raise NotImplementedError(f"LLM_PROVIDER '{LLM_PROVIDER}' is not supported. Use 'local' or 'openrouter'.")

# Embedding/Reranking provider: "local" (default, huggingface) or "openrouter"
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")

if EMBEDDING_PROVIDER not in ["local", "openrouter"]:
    raise NotImplementedError(f"EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}' is not supported. Use 'local' or 'openrouter'.")


# =============================================================================
# MODEL CONFIGURATION - LOCAL
# =============================================================================
# LLM Settings (Local - Ollama)
OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Local Embedding & Reranking Settings (used when EMBEDDING_PROVIDER=local)
LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
LOCAL_RERANKER_MODEL: str = os.getenv("LOCAL_RERANKER_MODEL", "BAAI/bge-reranker-base")


# =============================================================================
# MODEL CONFIGURATION - OPENROUTER
# =============================================================================
# General Settings
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.5-9b")
OPENROUTER_AGENT_MODEL: str = os.getenv("OPENROUTER_AGENT_MODEL", "qwen/qwen3-32b")

# Embedding Model Settings
OPENROUTER_EMBEDDING_MODEL: str = os.getenv("OPENROUTER_EMBEDDING_MODEL", "baai/bge-m3")
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

# Reranker Model Settings
OPENROUTER_RERANKER_MODEL: str = os.getenv("OPENROUTER_RERANKER_MODEL", "cohere/rerank-v3.5")


# =============================================================================
# RAG & VECTOR DATABASE CONFIGURATION
# =============================================================================
COLLECTION_NAME: str = "diem_collect_HeaderContext_Nuova_Versione"
DEFAULT_SESSION_ID: str = "diem-session"

# Retrieval Settings
# BI_ENCODER_K: number of documents retrieved in the first stage (fast retrieval)
BI_ENCODER_K: int = int(os.getenv("BI_ENCODER_K", "20"))
# CROSS_ENCODER_K: number of documents kept after reranking in the second stage (precision reranking)
CROSS_ENCODER_K: int = int(os.getenv("CROSS_ENCODER_K", "3"))

RETRIEVER_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.7"))

# Document Splitting Settings (Parent-Child Strategy)
# Parent Document Settings (Broad context)
# TODO (Software Architect): Consider making chunk sizes configurable via environment variables.
PARENT_CHUNK_SIZE: int = 2000
PARENT_CHUNK_OVERLAP: int = 200

# Child Document Settings (Precise retrieval)
CHILD_CHUNK_SIZE: int = 400
CHILD_CHUNK_OVERLAP: int = 50

# Ingestion Batching
MAX_CHILD_CHUNKS_PER_BATCH: int = 4000


# =============================================================================
# AGENTIC RAG & ENRICHMENT CONFIGURATION
# =============================================================================
# Agentic RAG Settings
# Max tool calls the 32b agent can make per turn before generate is forced
MAX_TOOL_CALLS: int = int(os.getenv("MAX_TOOL_CALLS", "6"))

# OpenRouter Enrichment Settings
OPENROUTER_ENDPOINT: str = os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_CONTEXT_HEADER_MODEL: str = os.getenv("OPENROUTER_CONTEXT_HEADER_MODEL", "mistralai/mistral-nemo")
OPENROUTER_TIMEOUT_SECONDS: float = float(os.getenv("OPENROUTER_CONTEXT_HEADER_TIMEOUT", "30"))
MAX_OPENROUTER_FAILURES: int = int(os.getenv("OPENROUTER_CONTEXT_HEADER_MAX_FAILURES", "3"))

# Ollama Enrichment Settings
OLLAMA_ENDPOINT: str = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL: str = os.getenv("OLLAMA_ENRICHMENT_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_ENRICHMENT_TIMEOUT", "10"))
MAX_OLLAMA_FAILURES: int = int(os.getenv("OLLAMA_ENRICHMENT_MAX_FAILURES", "5"))
