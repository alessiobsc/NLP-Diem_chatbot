"""Automated test runner for DiemBrain fixes."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
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
    # ── A: "Sei sicuro?" challenge (target del fix) ─────────────────────────
    # A1: single challenge, fatto puntuale (replica rob_001)
    ("A1", "Dove si trova il DIEM?"),
    ("A1", "Sei sicuro?"),

    # A2: single challenge, persona
    ("A2", "Chi è il direttore del DIEM?"),
    ("A2", "Ne sei certo?"),

    # A3: single challenge, dato numerico (CFU tesi — dato era sbagliato in C4)
    ("A3", "Quanti CFU vale la tesi di laurea magistrale al DIEM?"),
    ("A3", "Sei sicuro di quello che hai detto?"),

    # A4: double challenge su lista (replica rob_005)
    ("A4", "Quali corsi di laurea sono offerti dal DIEM?"),
    ("A4", "Sei davvero sicuro di quella lista?"),
    ("A4", "Penso che ne hai dimenticato uno. Puoi rielencarli?"),

    # A5: challenge su lista estesa (laboratori)
    ("A5", "Quali laboratori di ricerca ci sono al DIEM?"),
    ("A5", "Sicuro? Non mi sembra completo."),

    # A6: challenge molto corta (una parola)
    ("A6", "Chi è il delegato alla mobilità internazionale?"),
    ("A6", "Davvero?"),

    # A7: challenge su dato temporale
    ("A7", "Quando riceve il Professor Capuano?"),
    ("A7", "Sei sicuro che sia quel giorno?"),

    # ── B: Regression (il fix NON deve rompere questi) ──────────────────────
    # B1: richiesta legittima di chiarimento — deve ancora chiedere
    ("B1", "Quali corsi ci sono al DIEM?"),
    ("B1", "Quello più recente attivato"),

    # B2: USER DISSATISFIED (Rule 7b) — deve ri-retrievare, non chiedere
    ("B2", "Quali sono gli orari di ricevimento del Professor Capuano?"),
    ("B2", "Non hai risposto, ripeti"),

    # B3: FALSE PREMISE (Rule 8) — deve negare, non confermare
    ("B3", "Dove si trova il DIEM?"),
    ("B3", "Prima mi avevi detto che era a Napoli, puoi confermare?"),
]

for session, q in tests:
    print(f"[{session}] Q: {q}", flush=True)
    try:
        ans = brain.chat(q, session_id=session)
        print(f"[{session}] A: {ans[:600]}", flush=True)
    except Exception as e:
        print(f"[{session}] ERROR: {e}", flush=True)
    print(flush=True)
