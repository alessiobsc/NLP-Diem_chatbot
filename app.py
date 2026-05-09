import os
import sys
os.environ.setdefault("PYTHONUNBUFFERED", "1")
from dotenv import load_dotenv
from langchain_chroma import Chroma
from brain import embedding_model, DiemBrain
import gradio as gr

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_DIR    = "chroma_diem"
COLLECTION    = "diem_knowledge"
SESSION_ID    = "diem-session"
FORCE_REINDEX = "--reindex" in sys.argv

# ─────────────────────────────────────────────────────────────────────────────
# Vector store: load existing or build from scratch
# ─────────────────────────────────────────────────────────────────────────────
db_file = os.path.join(CHROMA_DIR, "chroma.sqlite3")

if FORCE_REINDEX or not os.path.exists(db_file):
    from main_ingestion import run_full_pipeline
    run_full_pipeline(embedding_model)
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )
else:
    print("Loading existing Chroma index...")
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
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
def chat_fn(message: str, history: list) -> str:
    return brain.chat(message, SESSION_ID)


demo = gr.ChatInterface(
    fn=chat_fn,
    title="DIEM Chatbot",
    description=(
        "Ask questions about the DIEM department (University of Salerno): "
        "degree programs, faculty, research, courses, regulations, and more."
    ),
    examples=[
        "What degree programs are offered by DIEM?",
        "Where is DIEM located?",
        "What research areas are active at DIEM?",
        "Who is responsible for internationalization at DIEM?",
        "Which laboratories are available at DIEM?",
    ],
    chatbot=gr.Chatbot(height=500),
)

if __name__ == "__main__":
    demo.launch()
