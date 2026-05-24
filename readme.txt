================================================================================
DIEM CHATBOT - README
================================================================================
Chatbot conversazionale Retrieval-Augmented Generation (RAG) per il
Dipartimento di Ingegneria dell'Informazione ed Elettrica e Matematica
applicata (DIEM) dell'Universita degli Studi di Salerno.

Il sistema effettua il crawl del sito DIEM e delle pagine docenti/corsi
correlate, indicizza i contenuti in un vector store Qdrant, e serve una UI
Gradio che risponde a domande in linguaggio naturale citando le fonti.

================================================================================
1. STRUTTURA DELLE DIRECTORY
================================================================================

NLP-Diem_chatbot/
|
+-- app.py                      # Entry point UI (Gradio + caricamento DB)
+-- main_ingestion.py           # Entry point della pipeline di ingestion
+-- config.py                   # Configurazione centralizzata (Singleton)
+-- requirements.txt            # Dipendenze Python
+-- .env                        # Variabili d'ambiente (NON in git)
+-- .gitignore
+-- crawled_urls.json           # Snapshot URL HTML crawlati (output ingestion)
+-- crawled_pdfs.json           # Snapshot URL PDF crawlati (output ingestion)
|
+-- src/
|   +-- agent/
|       +-- brain.py            # Core RAG: agent, tools, graph
|       +-- state.py            # State definition for the LangGraph agent
|       +-- nodes.py            # Node implementations for the graph
|       +-- tools.py            # RAG tools (retrieve, rewrite, etc.)
|   +-- rag_hybrid.py           # Logica per RAG Ibrido con Qdrant e BGE-M3
|   +-- prompts.py              # Repository centralizzato dei prompt
|   +-- utils/
|       +-- logger.py           # Logger condiviso (console + file rotante)
|   +-- ingestion/
|       +-- crawler.py          # Crawling siti DIEM/docenti/corsi
|       +-- parser.py           # Estrazione testo HTML, metadati, PDF
|       +-- enrichment.py       # Generazione context header (Ollama)
|       +-- database.py         # Splitting e indicizzazione in Qdrant
|
+-- scripts/                    # Tool di debug e ispezione (non usati a runtime)
|   +-- check_vector_db.py
|   +-- test_chunks.py
|   +-- test_retriever.py
|
+-- evaluation/                 # Framework di valutazione del chatbot
|   +-- ...
|
+-- logs/                       # (generata) log applicativi a rotazione
|   +-- chatbot.log + chatbot.log.1..N

================================================================================
2. CONFIGURAZIONE - config.py
================================================================================
Modulo Singleton che concentra tutte le configurazioni. Carica `.env`.

Costanti principali:
- PROJECT_ROOT: path assoluto del progetto.
- LOG_LEVEL, LOG_DIR, LOG_FILE, MAX_LOG_SIZE_MB, LOG_BACKUP_COUNT.
- LLM_PROVIDER (local o openrouter).
- OLLAMA_CHAT_MODEL (default qwen2.5:7b), LLM_TEMPERATURE (0.1).
- LOCAL_EMBEDDING_MODEL (BAAI/bge-m3).
- QDRANT_HOST, QDRANT_PORT.
- COLLECTION_NAME ("hybrid_docs"), DEFAULT_SESSION_ID.
- CHILD_CHUNK_SIZE=300, CHILD_CHUNK_OVERLAP=80.

================================================================================
3. ENTRY POINT
================================================================================

--------------------------------------------------------------------------------
3.1 app.py - UI Gradio
--------------------------------------------------------------------------------
Cosa fa:
  - Carica `.env`.
  - Se e' passato `--reindex`, avvia l'intera pipeline di ingestion.
  - Altrimenti, si connette a un'istanza Qdrant esistente.
  - Istanzia `QdrantRAG` e `DiemBrain(hybrid_rag)`.
  - Avvia una `gr.ChatInterface` collegata a `brain.chat_stream`.

Input:        nessun argomento (flag opzionale `--reindex`).
Output:       UI Gradio sulla porta predefinita; log su stdout e `logs/chatbot.log`.

--------------------------------------------------------------------------------
3.2 main_ingestion.py - Pipeline di ingestion
--------------------------------------------------------------------------------
Cosa fa: orchestra le fasi di costruzione del knowledge base per Qdrant.

  Fase 1 - Crawl: (Invariato)
  Fase 2 - Filtraggio + enrichment: (Invariato)
  Fase 3 - Indicizzazione (`index_documents` -> `database.DocumentIndexer`):
    - Inizializza `QdrantRAG` e ricrea la collezione Qdrant.
    - Splitta i documenti in chunk.
    - Per ogni chunk, genera embedding (denso, sparso, colbert) con BGE-M3.
    - Esegue l'upsert del punto in Qdrant con i 3 vettori e il payload.

Modalita CLI:
  python main_ingestion.py --crawl-only    # solo Fase 1 + save JSON
  python main_ingestion.py --full          # pipeline completa

================================================================================
4. MODULO src/
================================================================================

--------------------------------------------------------------------------------
4.1 src/rag_hybrid.py
--------------------------------------------------------------------------------
Cuore del sistema di retrieval ibrido.

Classe `QdrantRAG`:
  - Gestisce la connessione a Qdrant e il modello BGE-M3.
  - `create_collection`: configura la collezione `hybrid_docs` con 3 vettori nominati (dense, sparse, colbert).
  - `generate_embeddings`: genera i 3 tipi di embedding in un'unica chiamata a BGE-M3.
  - `upsert_document`: indicizza un documento in Qdrant.
  - `search`: esegue una ricerca ibrida usando Reciprocal Rank Fusion (RRF) per combinare i risultati.

--------------------------------------------------------------------------------
4.2 src/agent/brain.py
--------------------------------------------------------------------------------
Logica principale dell'agente basato su LangGraph.

Classe `DiemBrain`:
  - Costruita su `QdrantRAG`.
  - `_build_graph`: costruisce lo StateGraph con nodi per input/output guards, scope check, e chiamate agli strumenti.
  - La tool `retrieve` ora chiama `hybrid_rag.search()`.
  - `chat_stream`: orchestra l'esecuzione del grafo e fa lo streaming della risposta.

================================================================================
5. SCRIPT DI DEBUG - scripts/
================================================================================

scripts/check_vector_db.py
  Si connette a Qdrant e ispeziona i chunk nella collezione `hybrid_docs`,
  verificando la propagazione degli header, la distribuzione per dominio e la qualità del contenuto.

scripts/test_chunks.py
  Mostra statistiche dettagliate e campioni dei chunk presenti in Qdrant.

scripts/test_retriever.py
  REPL interattiva: l'utente immette una query, lo script esegue una ricerca
  ibrida tramite `QdrantRAG` e mostra i risultati con score e metadati.

================================================================================
6. LIBRERIE USATE (requirements.txt)
================================================================================

Core RAG (LangChain 1.x):
  langchain, langchain-core, langchain-community, etc.

Vector DB:
  qdrant-client

UI:
  gradio

Document processing:
  pdfplumber, beautifulsoup4, trafilatura, etc.

HuggingFace / ML:
  transformers, sentence-transformers, FlagEmbedding, torch, numpy.

Utility:
  python-dotenv

Dipendenze esterne (NON in requirements):
  - Server Qdrant in locale o remoto.
  - Server Ollama in locale per LLM e arricchimento.

================================================================================
7. FLUSSO COMPLETO DI ESECUZIONE
================================================================================

A) Setup
   python -m venv venv && venv\Scripts\activate
   pip install -r requirements.txt
   # Avviare i container Docker per Qdrant e Ollama
   docker run -p 6333:6333 qdrant/qdrant
   # Avviare Ollama e pullare i modelli necessari

B) Ingestion (una-tantum o on-demand)
   python main_ingestion.py --full

C) Avvio UI
   python app.py
   python app.py --reindex

D) Debug
   python scripts/check_vector_db.py
   python scripts/test_chunks.py
   python scripts/test_retriever.py

================================================================================
8. VARIABILI D'AMBIENTE (.env)
================================================================================
  LOG_LEVEL
  OLLAMA_CHAT_MODEL, LLM_TEMPERATURE
  QDRANT_HOST, QDRANT_PORT
  OPENROUTER_API_KEY (opzionale)

================================================================================
