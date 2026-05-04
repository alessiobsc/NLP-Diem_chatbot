# DIEM Chatbot — NLP Project

Chatbot RAG per il sito del DIEM (Università di Salerno). Esame NLP, gruppo di 3-4 studenti.

## Architettura attuale

```
Web crawling  →  Chunking  →  Embedding  →  Chroma DB (persistito)
                                                    ↓
User query  →  Query Rewriting  →  Retriever (top-5)  →  LLM  →  Risposta
                    ↑ history                                ↑ history
                    └──────────── RunnableWithMessageHistory ┘
```

### Componenti

| Componente | Scelta | Motivo |
|---|---|---|
| LLM | `Qwen/Qwen2.5-7B-Instruct` via HuggingFace Endpoint | Gratuito, instruct-tuned |
| Embedding | `BAAI/bge-small-en-v1.5` (locale) | Leggero, buona qualità similarity |
| Vector store | Chroma (persistito in `chroma_diem/`) | Dal corso, no re-index a ogni avvio |
| Chunking | `RecursiveCharacterTextSplitter` (1000 char, 150 overlap) | Robusto su HTML eterogeneo |
| UI | Gradio 6 (`ChatInterface`) | Demo live per esame |

### Crawling

- `diem.unisa.it` — crawl raw (depth 3), estrae link a docenti/corsi
- `docenti.unisa.it` — pagine trovate via link (cap 50 docenti)
- `corsi.unisa.it` — pagine trovate via link (cap 30 corsi)
- Extractor BeautifulSoup: rimuove nav/footer/script, mantiene contenuto
- **515 documenti → 11134 chunk** indicizzati

### RAG Chain (da `13. RAG (exercise).ipynb`)

1. **Query rewriting** — risolve pronomi/riferimenti usando history
2. **Retrieval** — top-5 chunk per similarità coseno
3. **Generation** — prompt con scope awareness + anti-hallucination
4. **History** — `RunnableWithMessageHistory` + `InMemoryChatMessageHistory`

### System prompt

- Risponde SOLO da contesto (no prior knowledge)
- Out-of-domain → risposta esplicita di rifiuto
- Anti-hallucination esplicita

### Avvio

```bash
# Prima volta (crawla e indicizza ~20 min):
venv/Scripts/python -u diem-chatbot.py

# Avvii successivi (carica Chroma da disco, ~30 sec):
venv/Scripts/python -u diem-chatbot.py

# Forzare re-index:
venv/Scripts/python -u diem-chatbot.py --reindex
```

Gradio su `http://127.0.0.1:7860` (porta può variare se occupata).

---

## Prossimi passi

### 1. Reranking (da `14. Advanced RAG Techniques.ipynb`)

Usare `CrossEncoder` dopo il retrieval per riordinare i chunk.

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
# retriever k=20, poi rerank → top 5
```

### 2. Chunking avanzato (da notebook 14)

- **Semantic chunking** (`SemanticChunker`) — split su rotture semantiche reali
- **Parent-child chunking** (`ParentDocumentRetriever`) — chunk piccoli per retrieval, chunk grandi per contesto LLM

### 3. PDF loader

Caricare PDF collegati alle pagine DIEM (regolamenti, bandi, manifesto studi).
- `PyPDFLoader` / `PyPDFDirectoryLoader`
- Estrarre link `.pdf` durante crawling e caricarli separatamente

### 4. Più docenti crawlati

Attualmente solo 5 docenti trovati via link da diem.unisa.it.
- Trovare endpoint diretto lista docenti DIEM
- Caricare tutte le pagine: home, ricevimento, insegnamenti, pubblicazioni

### 5. RAG Evaluation (da `15. RAG Evaluation.ipynb`)

Framework **RAGAS** con dataset di domande campione dal progetto.

```python
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    ContextPrecision, ContextRecall,
    ResponseRelevancy, Faithfulness, FactualCorrectness
)
```

Metriche da misurare:
- `ContextPrecision` — chunk recuperati rilevanti?
- `ContextRecall` — risposta coperta dal contesto?
- `ResponseRelevancy` — risposta pertinente alla domanda?
- `Faithfulness` — risposta fedele al contesto (no hallucination)?
- `FactualCorrectness` (F1) — correttezza fattuale vs reference

Dataset: lista domande campione dal progetto (pag. 4-5 del PDF) + risposte attese manuali.

---

## Criteri di valutazione esame

- **70%** performance chatbot: Relevance, Correctness, Coherence, Robustness, Scope Awareness
- **30%** design + report + UX + demo (max 5 min)

## File principali

```
diem-chatbot.py          # script unico, tutto il pipeline
chroma_diem/             # vector store persistito (non committare)
CLAUDE.md                # questo file
```
