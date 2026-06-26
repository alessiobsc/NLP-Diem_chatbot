"""Automated test runner for DiemBrain fixes."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv; load_dotenv()

from langchain_chroma import Chroma
from config import CHROMA_DIR_NAME, COLLECTION_NAME, EMBEDDING_DIMENSION
from src.encoders.embedding_init import build_embedding_model
from src.agent.brain import DiemBrain

import argparse
_args = argparse.ArgumentParser()
_args.add_argument("--calc-only", action="store_true", help="Skip robustness tests, run only calculate comparison")
_args = _args.parse_args()

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

if not _args.calc_only:
    for session, q in tests:
        print(f"[{session}] Q: {q}", flush=True)
        try:
            ans = brain.chat(q, session_id=session)
            print(f"[{session}] A: {ans[:600]}", flush=True)
        except Exception as e:
            print(f"[{session}] ERROR: {e}", flush=True)
        print(flush=True)

# ── Calculate tool test ───────────────────────────────────────────────────────
import time
from src.agent.brain import STREAM_DEGENERATE_SIGNAL

calc_questions = [
    ("CALC_MAG_INF",
     "Con media 27 quanto posso prendere alla laurea magistrale in Ingegneria Informatica?"),
    ("CALC_TRI_INF",
     "Con media 27 quanto posso prendere alla laurea triennale in Ingegneria Informatica?"),
    ("CALC_MAG_IEDM",
     "Con media 27 quanto posso prendere alla laurea magistrale in "
     "Information Engineering for Digital Medicine?"),
]


def run_timed(b, q, sid):
    """Stream chat; return (answer, ttft_ms, total_ms)."""
    t0 = time.perf_counter()
    t_first = None
    chunks = []
    for chunk in b.chat_stream(q, session_id=sid):
        if t_first is None:
            t_first = time.perf_counter()
        chunks.append(chunk)
    t_end = time.perf_counter()
    ans = "".join(chunks)
    ttft  = round((t_first - t0) * 1000) if t_first else None
    total = round((t_end - t0) * 1000)
    return ans, ttft, total


print(f"\n{'='*60}", flush=True)
print("CALCULATE TEST", flush=True)
print(f"{'='*60}", flush=True)
for sid, q in calc_questions:
    print(f"\n[{sid}] Q: {q}", flush=True)
    try:
        ans, ttft, total = run_timed(brain, q, sid)
        print(f"[{sid}] TTFT: {ttft}ms | Total: {total}ms", flush=True)
        if ans == STREAM_DEGENERATE_SIGNAL:
            print(f"[{sid}] A: <DEGENERATE>", flush=True)
        else:
            print(f"[{sid}] A: {ans[:900]}", flush=True)
    except Exception as e:
        print(f"[{sid}] ERROR: {e}", flush=True)
    print(flush=True)
