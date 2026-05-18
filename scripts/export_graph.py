"""Export the DiemBrain LangGraph as a PNG image."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from langchain_chroma import Chroma
from config import CHROMA_DIR_NAME, COLLECTION_NAME, EMBEDDING_DIMENSION
from src.encoders.embedding_init import build_embedding_model
from src.agent.brain import DiemBrain

load_dotenv()

embedding_model = build_embedding_model()
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=CHROMA_DIR_NAME,
    collection_metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIMENSION},
)

brain = DiemBrain(vectorstore)
png_bytes = brain._graph.get_graph().draw_mermaid_png()

output_path = os.path.join("docs", "graph.png")
os.makedirs("docs", exist_ok=True)
with open(output_path, "wb") as f:
    f.write(png_bytes)

print(f"Graph saved to {output_path}")