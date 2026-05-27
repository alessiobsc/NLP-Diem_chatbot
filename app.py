import os
import sys
from config import CHROMA_DIR_NAME, COLLECTION_NAME, DEFAULT_SESSION_ID, EMBEDDING_DIMENSION
from dotenv import load_dotenv
from langchain_chroma import Chroma
import gradio as gr

from src.agent.brain import DiemBrain, STREAM_DEGENERATE_SIGNAL
from src.encoders.embedding_init import build_embedding_model
from src.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


# Global embedding model instance
embedding_model = build_embedding_model()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
# TODO (Code Refactorer): Use argparse instead of checking sys.argv directly for better CLI handling.
FORCE_REINDEX = "--reindex" in sys.argv

# ─────────────────────────────────────────────────────────────────────────────
# Vector store: load existing or build from scratch
# ─────────────────────────────────────────────────────────────────────────────
db_file = os.path.join(CHROMA_DIR_NAME, "chroma.sqlite3")

if FORCE_REINDEX or not os.path.exists(db_file):
    logger.info("Building Chroma index from scratch or forced reindex...")
    from main_ingestion import run_full_pipeline
    run_full_pipeline(embedding_model)
else:
    logger.info("Loading existing Chroma index...")

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=CHROMA_DIR_NAME,
    collection_metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIMENSION}
)

try:
    logger.info(f"  -> {vectorstore._collection.count()} chunks in index")
except Exception as e:
    logger.warning(f"  -> Could not count chunks in index: {e}")
    logger.info("  -> Index loaded")

# ─────────────────────────────────────────────────────────────────────────────
# Brain
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Initializing DiemBrain from app.py")
brain = DiemBrain(vectorstore)

# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
_DEGENERATE_FALLBACK = "Mi dispiace, non sono riuscito a trovare informazioni sufficienti per rispondere a questa domanda."

def chat_fn(message: str, history: list):
    accumulated = ""
    emitted = False
    try:
        for chunk in brain.chat_stream(message, DEFAULT_SESSION_ID):
            if chunk == STREAM_DEGENERATE_SIGNAL:
                # Answer was degenerate (<30 chars): replace entire display with fallback.
                yield _DEGENERATE_FALLBACK
                return
            if not chunk:
                continue
            accumulated += chunk
            emitted = True
            yield accumulated
    except Exception as e:
        logger.exception(f"Gradio chat stream error: {e}")
        yield "Mi dispiace, si è verificato un errore."
        return

    if not emitted:
        yield "Mi dispiace, non sono riuscito a generare una risposta."


demo = gr.ChatInterface(
    fn=chat_fn,
    title="DIEM Chatbot",
    description=(
        "Ask questions about the DIEM department (University of Salerno): "
        "degree programs, faculty, research, courses, regulations, and more."
    ),
    examples=[
        "Quali corsi di laurea offre il DIEM?",
        "Dove si trova il DIEM?",
        "Quali sono le aree di ricerca attive al DIEM?",
        "Chi è responsabile dell'internazionalizzazione al DIEM?",
        "Quali laboratori sono disponibili al DIEM?",
        "Ho preso 18 al TOLC. Posso iscrivermi?",
        "Qual è il programma del corso di Ingegneria del Software?",
        "Quali sono gli orari di ricevimento del professore Capuano?"
    ],
    chatbot=gr.Chatbot(
        height=500,
        latex_delimiters=[
            {"left": "$$", "right": "$$", "display": True},
            {"left": "$", "right": "$", "display": False},
            {"left": "\\(", "right": "\\)", "display": False},
            {"left": "\\[", "right": "\\]", "display": True},
        ],
    ),
)

if __name__ == "__main__":
    logger.info("Launching Gradio interface")
    demo.launch()
