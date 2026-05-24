import sys
from config import DEFAULT_SESSION_ID
from dotenv import load_dotenv
import gradio as gr

from src.agent.brain import DiemBrain, STREAM_DEGENERATE_SIGNAL
from src.rag_hybrid import QdrantRAG
from src.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
FORCE_REINDEX = "--reindex" in sys.argv
# Use in-memory Qdrant by default, unless --reindex is passed which implies persistence
USE_IN_MEMORY_DB = not FORCE_REINDEX

# ─────────────────────────────────────────────────────────────────────────────
# Vector store: load existing or build from scratch
# ─────────────────────────────────────────────────────────────────────────────
# Initialize QdrantRAG
qdrant_rag = QdrantRAG(in_memory=USE_IN_MEMORY_DB)

if FORCE_REINDEX:
    logger.info("Building Qdrant index from scratch (forced reindex)...")
    from main_ingestion import run_full_pipeline
    run_full_pipeline(in_memory=False)
else:
    logger.info("Using in-memory Qdrant instance.")


# ─────────────────────────────────────────────────────────────────────────────
# Brain
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Initializing DiemBrain from app.py")
brain = DiemBrain(qdrant_rag)

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
    chatbot=gr.Chatbot(height=500),
)

if __name__ == "__main__":
    logger.info("Launching Gradio interface")
    demo.launch()
