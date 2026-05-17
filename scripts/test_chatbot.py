"""Quick interactive test for DiemBrain without Gradio or LangGraph Studio."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from config import CHROMA_DIR_NAME, COLLECTION_NAME, EMBEDDING_DIMENSION
from src.encoders.embedding_init import build_embedding_model
from src.agent.brain import DiemBrain

embedding_model = build_embedding_model()
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=CHROMA_DIR_NAME,
    collection_metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIMENSION},
)
brain = DiemBrain(vectorstore)

session = "quicktest"
print("DiemBrain ready. Empty input to quit.\n")
while True:
    q = input("Q: ").strip()
    if not q:
        break
    print(f"A: {brain.chat(q, session_id=session)}\n")
