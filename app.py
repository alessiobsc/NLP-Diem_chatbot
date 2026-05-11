import argparse
import os
import shutil

import gradio as gr
from langchain_chroma import Chroma

from config import CHROMA_DIR_NAME, COLLECTION_NAME, DEFAULT_SESSION_ID
from main_ingestion import run_full_pipeline
from src.brain import embedding_model, DiemBrain
from src.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="DIEM Chatbot - Gradio Interface")
    parser.add_argument("--full", action="store_true", help="Force run the entire ingestion pipeline and rebuild the DB")
    args = parser.parse_args()

    db_file = os.path.join(CHROMA_DIR_NAME, "chroma.sqlite3")

    if args.full or not os.path.exists(db_file):
        logger.info("Building Chroma index from scratch or forced reindex...")
        if os.path.exists(CHROMA_DIR_NAME):
            shutil.rmtree(CHROMA_DIR_NAME)
        run_full_pipeline(embedding_model)
    else:
        logger.info("Loading existing Chroma index...")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    brain = DiemBrain(vectorstore)

    def chat_interface(message, history):
        """
        Gradio calls this function with the new message and the history.
        We stream the response using brain.chat_stream.
        """
        for partial_response in brain.chat_stream(message, session_id=DEFAULT_SESSION_ID):
            yield partial_response

    demo = gr.ChatInterface(
        fn=chat_interface,
        title="DIEM Department Assistant",
        description=(
            "Ask questions about the DIEM department (University of Salerno): "
            "degree programs, faculty, research, courses, regulations, and more."
        ),
        theme="soft",
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
    )

    demo.launch()

if __name__ == "__main__":
    main()
