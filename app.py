import os
import sys
from config import CHROMA_DIR_NAME, COLLECTION_NAME, DEFAULT_SESSION_ID
from dotenv import load_dotenv
from langchain_chroma import Chroma
from brain import embedding_model, DiemBrain
import gradio as gr

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
FORCE_REINDEX = "--reindex" in sys.argv

# ─────────────────────────────────────────────────────────────────────────────
# Vector store: load existing or build from scratch
# ─────────────────────────────────────────────────────────────────────────────
db_file = os.path.join(CHROMA_DIR_NAME, "chroma.sqlite3")

if FORCE_REINDEX or not os.path.exists(db_file):
    from main_ingestion import run_full_pipeline
    run_full_pipeline(embedding_model)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR_NAME,
    )
else:
    print("Loading existing Chroma index...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR_NAME,
    )
    try:
        print(f"  -> {vectorstore._collection.count()} chunks in index")
    except Exception:
        print("  -> Index loaded")

# ─────────────────────────────────────────────────────────────────────────────
# Brain
# ─────────────────────────────────────────────────────────────────────────────
brain = DiemBrain(vectorstore)

# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
def chat_fn(message: str, history: list):
    yield from brain.chat_stream(message, DEFAULT_SESSION_ID)


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
    demo.launch()
