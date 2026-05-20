# DIEM Chatbot — Agentic RAG

> **NLP & Large Language Models** · Corso di Laurea Magistrale in Ingegneria Informatica  
> DIEM – Università degli Studi di Salerno

An LLM-powered conversational chatbot that answers questions about the DIEM department (University of Salerno) using **Agentic Retrieval-Augmented Generation** over a knowledge base built from the official DIEM websites.

---

## What it does

Users can ask natural language questions about DIEM in Italian or English:

| User type | Example questions |
|-----------|------------------|
| Prospective students | *What degree programs does DIEM offer? What are the admission requirements?* |
| Current students | *What are Prof. Capuano's office hours? Where is Room 126?* |
| International students | *What Erasmus opportunities are available at DIEM?* |
| Researchers / visitors | *Which laboratories are active? What is the DIEM research focus?* |

The chatbot **cites sources**, **rejects out-of-scope questions**, and **resists adversarial prompts**.

---

## Architecture

### Agentic RAG Graph (`diem_rag_graph`)

```
input_guard → scope_guard → reset_state → agent ↔ tools → output_guard → END
                                                       ↘ force_answer ↗
                                              [forced_retrieve if agent skips]
```

| Node | Role |
|------|------|
| `input_guard` | Regex-based offensive content filter on user input |
| `scope_guard` | Keyword fast-path + lightweight LLM on ambiguous queries; fail-open |
| `reset_state` | Clears per-turn state (tool call count, context, docs) |
| `agent` | 120B model: decides tool calls or generates final answer; dynamic hint injection |
| `tools` | Executes tool calls; counts only `retrieve` toward cap |
| `forced_retrieve` | Safety net if agent produces empty output without retrieving |
| `force_answer` | Fires if agent requests retrieve with `tool_call_count ≥ 3` |
| `output_guard` | Offensive content filter + PII redaction on final answer |

### Agent Tools

| Tool | Model | Purpose |
|------|-------|---------|
| `rewrite` | LLaMA 3.1 8B | Resolves pronouns, injects current year into standalone query |
| `retrieve` | — | Bi-encoder + cross-encoder retrieval from Chroma |
| `summarize` | LLaMA 3.1 8B | Condenses long retrieved context |
| `calculate` | — | Safe arithmetic evaluation for numeric questions |

### Retrieval Pipeline

```
Query → E5 bi-encoder (k=20) → Chroma child chunks
      → cross-encoder reranker → top-K parents → LLM
```

Parent-child chunking: parents (2000 / 200 overlap) stored in `LocalFileStore`; children (400 / 50 overlap) embedded and indexed in Chroma with context header prepended.

---

## Knowledge Base

| Source | Pages |
|--------|-------|
| `www.diem.unisa.it` | 69 |
| `docenti.unisa.it` (55 DIEM faculty) | 1,222 |
| `corsi.unisa.it` (DIEM degree programs) | 296 |
| **PDF documents** | 109 sources |

**Total:** 4,155 documents · 5,111 parent chunks · 23,563 child chunks  
**Collection:** `diem_collect_HeaderContext_Nuova_Versione` · Cutoff: 2020

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Agent LLM | `openai/gpt-oss-120b` via OpenRouter |
| Lightweight LLM | `meta-llama/llama-3.1-8b-instruct` via OpenRouter |
| Enrichment LLM (ingestion) | `llama3.2:1b` via Ollama (fallback: heuristic) |
| Embedding | `intfloat/multilingual-e5-base` (local, 384 dim) |
| Cross-encoder | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (local) |
| Vector store | Chroma (cosine HNSW) |
| Agent framework | LangGraph + LangChain |
| UI | Gradio 6 `ChatInterface` |
| Evaluation | Ragas + custom metrics |

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt
```

### Environment variables (`.env`)

```env
# Required
HF_TOKEN=your_huggingface_token
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_AGENT_MODEL=openai/gpt-oss-120b
OPENROUTER_LIGHTWEIGHT_MODEL=meta-llama/llama-3.1-8b-instruct

# Optional — LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=diem-chatbot

# Optional — Ollama (only needed for ingestion enrichment)
OLLAMA_ENDPOINT=http://localhost:11434
```

> With `LLM_PROVIDER=openrouter`, Ollama is **not required** to run the chatbot. It is only used during the ingestion enrichment phase to generate context headers.

---

## Usage

### Build the knowledge base

```bash
# Crawl only (saves crawled_urls.json and crawled_pdfs.json)
venv/Scripts/python -u main_ingestion.py --crawl-only

# Full pipeline: crawl → filter → enrich → embed → index
venv/Scripts/python -u main_ingestion.py --full
```

### Run the chatbot

```bash
# Gradio web UI (requires chroma_diem/ to exist)
venv/Scripts/python -u app.py

# Interactive CLI test
venv/Scripts/python scripts/test_chatbot.py

# Automated test (non-interactive)
venv/Scripts/python scripts/test_chatbot_auto.py
```

### Evaluation

```bash
# Full evaluation suite (Italian)
venv/Scripts/python evaluation/tester.py --lang it

# With cache and limited questions
venv/Scripts/python evaluation/tester.py --lang it --limit 5 --cache use

# English golden set
venv/Scripts/python evaluation/tester.py --lang en
```

---

## Project Structure

```
├── app.py                          # Gradio UI entry point
├── main_ingestion.py               # Ingestion pipeline entry point
├── config.py                       # Centralized configuration (singleton)
├── requirements.txt
│
├── src/
│   ├── agent/
│   │   ├── brain.py                # DiemBrain(DiemNodes) — graph build, public API
│   │   ├── nodes.py                # DiemNodes mixin — all _node_* graph nodes
│   │   ├── state.py                # DiemState TypedDict
│   │   ├── tools.py                # @tool: rewrite, retrieve, summarize, calculate
│   │   ├── init_models.py          # build_agent_model(), build_lightweight_model()
│   │   └── utils.py                # extract_text, format_context, rewrite_query
│   │
│   ├── ingestion/
│   │   ├── crawler.py              # Crawling, link extraction, faculty validation
│   │   ├── crawl_state.py          # Crawl state management
│   │   ├── parser.py               # HTML extractor, temporal filter, PDF loader
│   │   ├── enrichment.py           # Context headers via Ollama (fallback: heuristic)
│   │   └── database.py             # Parent-child split, embedding, Chroma index
│   │
│   ├── encoders/
│   │   ├── embedding_models.py     # E5 HuggingFace embedding wrapper
│   │   ├── embedding_init.py       # Shared embedding model instance
│   │   └── reranker.py             # Cross-encoder singleton
│   │
│   ├── utils/
│   │   └── logger.py               # Shared rotating file + console logger
│   │
│   ├── middleware.py               # ScopeGuardrail, OffensiveContentGuardrail, redact_pii
│   └── prompts.py                  # AGENT_SYSTEM_PROMPT, REWRITE_PROMPT, REJECTION_TAGS
│
├── evaluation/
│   ├── tester.py                   # Evaluation runner: Ragas + scope + robustness
│   ├── cache.py                    # Disk cache for chatbot responses
│   └── dataset/
│       ├── golden_set_it.json      # Italian golden set
│       └── golden_set_en.json      # English golden set
│
└── scripts/                        # Debug and inspection utilities
    ├── inspect_url.py              # Inspect extracted text from a single URL
    ├── test_chatbot.py             # Interactive CLI chatbot test
    ├── test_chatbot_auto.py        # Automated chatbot test
    ├── test_retriever.py           # REPL retriever inspector
    ├── test_chunks.py              # Chunk statistics viewer
    └── check_vector_db.py          # Verify context header propagation in Chroma
```

**Generated at runtime (gitignored):**

```
chroma_diem/          # Chroma vector store + parent LocalFileStore
logs/                 # Rotating application logs
evaluation/results/   # Per-run evaluation artifacts
evaluation/cache/     # Chatbot response cache
```

---

## Evaluation Criteria

The project is evaluated according to the course assignment:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Chatbot Performance** | **70%** | Qualitative evaluation of answers |
| ↳ Relevance | — | Does the answer address the query? |
| ↳ Correctness | — | Is the answer factually accurate and grounded? |
| ↳ Coherence | — | Logical flow and internal consistency |
| ↳ Robustness | — | Resistance to adversarial prompts ("Are you sure?") |
| ↳ Scope Awareness | — | Rejects out-of-domain questions correctly |
| **Design & Documentation** | **30%** | Architecture, report quality, UX, live demo |

Automated evaluation uses [Ragas](https://docs.ragas.io/) for in-scope metrics and custom LLM-as-judge for robustness and scope awareness, over golden sets in both Italian and English.

---

## Guardrails & Safety

- **Input guard**: regex filter for offensive content before any processing
- **Scope guard**: lightweight LLM classifier rejects out-of-domain queries; keyword fast-path for obvious cases; fail-open to avoid blocking legitimate questions
- **Output guard**: offensive content filter + PII redaction (email, phone, fiscal codes) on every response
- **Anti-hallucination**: agent grounded strictly on retrieved context; `[KNOWLEDGE_GAP]` tag when information is not in KB; `[FUORI_SCOPE]` tag for out-of-domain