"""Automated test runner for DiemBrain fixes."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv; load_dotenv()

from langchain_chroma import Chroma
from config import CHROMA_DIR_NAME, COLLECTION_NAME, EMBEDDING_DIMENSION
from src.encoders.embedding_init import build_embedding_model
from src.agent.brain import DiemBrain

print("Loading...", flush=True)
em = build_embedding_model()
vs = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=em,
    persist_directory=CHROMA_DIR_NAME,
    collection_metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIMENSION},
)
brain = DiemBrain(vs)
print("Ready.\n", flush=True)

tests = [
    ("s1", "Chi e Mario Vento?"),
    ("s1", "Quanti anni ha di esperienza?"),        # follow-up ambiguo
    ("s2", "Chi e Pippo Franco?"),                  # KNOWLEDGE_GAP
    ("s3", "Quali sono gli esami del terzo anno di informatica?"),  # rewrite + anno
    ("s4", "Qual e la capienza dell Aula De Candia?"),              # info non in KB
    ("s5", "Chi e Antonio Greco?"),                 # spesso dava vuoto
]

for session, q in tests:
    print(f"[{session}] Q: {q}", flush=True)
    try:
        ans = brain.chat(q, session_id=session)
        print(f"[{session}] A: {ans[:400]}", flush=True)
    except Exception as e:
        print(f"[{session}] ERROR: {e}", flush=True)
    print(flush=True)
