================================================================================
DIEM CHATBOT - README
================================================================================
Chatbot conversazionale Retrieval-Augmented Generation (RAG) per il
Dipartimento di Ingegneria dell'Informazione ed Elettrica e Matematica
applicata (DIEM) dell'Universita degli Studi di Salerno.

Il sistema effettua il crawl del sito DIEM e delle pagine docenti/corsi
correlate, indicizza i contenuti in un vector store Chroma, e serve una UI
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
|   +-- brain.py                # Core RAG: retriever, reranker, chain LLM
|   +-- prompts.py              # Repository centralizzato dei prompt
|   +-- logger.py               # Logger condiviso (console + file rotante)
|   +-- ingestion/
|       +-- crawler.py          # Crawling siti DIEM/docenti/corsi
|       +-- parser.py           # Estrazione testo HTML, metadati, PDF
|       +-- enrichment.py       # Generazione context header (Ollama)
|       +-- database.py         # Splitting parent/child + indicizzazione Chroma
|
+-- scripts/                    # Tool di debug e ispezione (non usati a runtime)
|   +-- check_vector_db.py
|   +-- test_chunks.py
|   +-- test_retriever.py
|
+-- evaluation/                 # Framework di valutazione del chatbot
|   +-- tester.py               # Runner principale: Ragas + scope + robustness
|   +-- cache.py                # Cache su disco per risposte del chatbot
|   +-- todo.txt                # TODO post-evaluation per la consegna
|   +-- dataset/
|   |   +-- golden_set_it.json  # Golden set in italiano
|   |   +-- golden_set_en.json  # Golden set in inglese
|   +-- results/                # Output per run timestampato (.gitkeep)
|       +-- <YYYYMMDD_HHMMSS>_<lang>/
|           +-- run.log
|           +-- per_question.json
|           +-- ragas_metrics.csv / .json
|           +-- scope_awareness.json
|           +-- robustness.json
|           +-- summary.md
|   +-- cache/                  # (generata) cache turni chatbot, gitignored
|
+-- chroma_diem/                # (generata) vector store Chroma, gitignored
|   +-- chroma.sqlite3          # SQLite con embeddings child
|   +-- <collection-uuid>/      # File HNSW di Chroma
|   +-- parent_store/           # Parent doc store (LocalFileStore key-value)
|
+-- logs/                       # (generata) log applicativi a rotazione
|   +-- chatbot.log + chatbot.log.1..N

================================================================================
2. CONFIGURAZIONE - config.py
================================================================================
Modulo Singleton (costanti a livello di modulo) che concentra tutte le
configurazioni dell'applicazione. Carica `.env` con python-dotenv.

Costanti principali:
- PROJECT_ROOT, CHROMA_DIR, PARENT_STORE_DIR: path assoluti del progetto.
- LOG_LEVEL, LOG_DIR, LOG_FILE, MAX_LOG_SIZE_MB, LOG_BACKUP_COUNT.
- OLLAMA_CHAT_MODEL (default qwen2.5:7b), LLM_TEMPERATURE (0.1).
- EMBEDDING_MODEL_NAME (intfloat/multilingual-e5-small).
- CROSS_ENCODER_MODEL_NAME (cross-encoder/mmarco-mMiniLMv2-L12-H384-v1).
- COLLECTION_NAME ("diem_collect_HeaderContext_new_Italiano"),
  DEFAULT_SESSION_ID.
- BI_ENCODER_K=20, CROSS_ENCODER_K=3, RETRIEVER_SCORE_THRESHOLD=0.7.
- PARENT_CHUNK_SIZE=2000/200, CHILD_CHUNK_SIZE=400/50 (overlap).
- MAX_CHILD_CHUNKS_PER_BATCH=4000 (limite batch in indicizzazione).

Tutti i moduli accedono a questi parametri via `from config import ...`.

================================================================================
3. ENTRY POINT
================================================================================

--------------------------------------------------------------------------------
3.1 app.py - UI Gradio
--------------------------------------------------------------------------------
Cosa fa:
  - Carica `.env`.
  - Se manca `chroma_diem/chroma.sqlite3` o e' passato `--reindex`, avvia
    l'intera pipeline di ingestion chiamando `main_ingestion.run_full_pipeline`.
  - Altrimenti apre la collezione Chroma esistente.
  - Istanzia `DiemBrain(vectorstore)` e una `gr.ChatInterface` collegata
    a `brain.chat_stream` (streaming token-per-token).

Input:        nessun argomento (flag opzionale `--reindex`).
Output:       UI Gradio sulla porta predefinita; log su stdout e `logs/chatbot.log`.
Si interfaccia con:
  - config.py             (path, modello, parametri)
  - src/brain.py          (DiemBrain, embedding_model)
  - src/logger.py         (logger condiviso)
  - main_ingestion.py     (solo se reindex)
  - langchain_chroma      (apertura vector store)
Librerie:     os, sys, dotenv, gradio, langchain_chroma.

--------------------------------------------------------------------------------
3.2 main_ingestion.py - Pipeline di ingestion
--------------------------------------------------------------------------------
Cosa fa: orchestra le 3 fasi del processo di costruzione del knowledge base.

  Fase 1 - Crawl (`crawl_phase`):
    1a. RecursiveUrlLoader su https://www.diem.unisa.it/ (depth 4).
    1b. Estrae lista docenti via `extract_diem_faculty_urls` -> crawla
        ciascuna pagina docente su docenti.unisa.it (depth 3) e le pagine
        annuali di pubblicazioni dal 2020 a oggi (depth 1).
    1c. Estrae URL corsi via `extract_corsi_urls` -> crawla i corsi su
        corsi.unisa.it (depth 3).
    Per ogni batch HTML invoca `load_pdfs_from_links` per scaricare i PDF
    linkati e li accumula separatamente.

  Fase 1b - Filtro metadata (`apply_html_metadata_and_filter`):
    Su ogni HTML grezzo applica `extract_html_metadata` (title, lang, date),
    sostituisce page_content con il testo estratto da `html_extractor`, e
    scarta le pagine con `lang` non italiano.

  Fase 2 - Filtraggio temporale + enrichment:
    `filter_recent_documents` scarta documenti con anno < 2020.
    `add_context_headers` chiama Ollama per generare un "context header" di
    una frase per ciascun documento.

  Fase 3 - Indicizzazione (`index_documents` -> `database.DocumentIndexer`):
    Split parent (2000/200) -> child (400/50, con header prepended) ->
    embedding multilingue E5 -> upsert in Chroma + persist parent in
    LocalFileStore.

Modalita CLI:
  python main_ingestion.py --crawl-only    # solo Fase 1 + save JSON
  python main_ingestion.py --full          # pipeline completa

Si interfaccia con:
  src/ingestion/crawler.py, parser.py, enrichment.py, database.py
  src/logger.py, config.py
  src/brain.py (per importare `embedding_model` quando lancia --full)
Output:
  - `crawled_urls.json`, `crawled_pdfs.json` (snapshot post-filtri)
  - `chroma_diem/` (vector store + parent_store popolati)
Librerie:     argparse, datetime, dotenv, langchain_community.

================================================================================
4. MODULO src/
================================================================================

--------------------------------------------------------------------------------
4.1 src/brain.py
--------------------------------------------------------------------------------
Cuore del sistema RAG.

Componenti chiave:
  - `E5HuggingFaceEmbeddings`: wrapper di HuggingFaceEmbeddings che antepone
    "passage: " durante l'ingestion e "query: " durante il retrieval, come
    richiesto dai modelli E5.
  - `embedding_model`: istanza globale condivisa con app.py e ingestion.
  - `reranker` (CrossEncoder): mMiniLMv2 multilingue, usato per il
    re-ranking di precisione.
  - `rerank(query, documents, top_n)`: scoring dei pair (query, content) e
    restituzione dei top-N con `relevance_score` in metadata.

Classe `DiemBrain`:
  - Costruita su `Chroma` vectorstore.
  - `_build_retriever` -> `ParentDocumentRetriever` con bi-encoder
    similarity_score_threshold (k=20, threshold=0.7) e split child 400/50.
  - `_build_rag_chain` -> chain LangChain:
      rewrite (condizionale alla presenza di history)
        -> retriever (bi-encoder)
        -> rerank (cross-encoder, top 3)
        -> _format_context (avvolge i doc in tag <document>/<source>)
        -> LLM (ChatOllama, qwen2.5:7b di default)
  - `conversational_rag`: `RunnableWithMessageHistory` con sessioni
    in-memory (`InMemoryChatMessageHistory`).
  - `chat(message, session_id)`: invocazione sincrona, ritorna risposta +
    elenco fonti.
  - `chat_stream(message, session_id)`: generatore che esegue
    silenziosamente la pipeline RAG e poi fa stream dei token del LLM;
    appende le fonti al termine (usato dalla UI Gradio).

Si interfaccia con: config.py, src/prompts.py, src/logger.py.
Librerie principali: langchain_classic (ParentDocumentRetriever, storage),
langchain_core (chat history, prompts, runnables), langchain_chroma,
langchain_huggingface, langchain_ollama, langchain_text_splitters,
sentence_transformers (CrossEncoder).

--------------------------------------------------------------------------------
4.2 src/prompts.py
--------------------------------------------------------------------------------
Tre prompt:
  - SYSTEM_PROMPT: identita assistant DIEM in inglese; impone risposta
    grounded sul contesto; due frasi standard per scope-rejection e
    knowledge-gap.
  - REWRITE_PROMPT: query rewriter (Italian/multilingual); riformula la
    domanda in standalone usando la history; vincoli anti-hallucination.
  - CONTEXT_HEADER_PROMPT: prompt italiano per generare l'header di 15
    parole massimo (usato da enrichment.py).

Si interfaccia con: src/brain.py (SYSTEM_PROMPT, REWRITE_PROMPT) e
src/ingestion/enrichment.py (CONTEXT_HEADER_PROMPT non importato
direttamente: enrichment usa un prompt inline analogo).

--------------------------------------------------------------------------------
4.3 src/logger.py
--------------------------------------------------------------------------------
`get_logger(name)` -> logger con:
  - Console handler (stdout).
  - RotatingFileHandler su `logs/chatbot.log`, max 10 MB, 5 backup.
  - Livello e dimensioni configurabili via env (LOG_LEVEL, MAX_LOG_SIZE_MB,
    LOG_BACKUP_COUNT).
Usato da TUTTI i moduli del progetto (app, ingestion, brain, scripts).

================================================================================
5. MODULO src/ingestion/
================================================================================

--------------------------------------------------------------------------------
5.1 crawler.py
--------------------------------------------------------------------------------
Cosa fa:
  - `crawl(start_url, base_url, max_depth)`: wrapper resiliente attorno a
    `RecursiveUrlLoader` di LangChain (regex HTML_LINK_REGEX + exclude
    di asset binari/static).
  - `extract_corsi_urls(raw_docs)`: estrae da /didattica/offerta-formativa
    i link a corsi.unisa.it (deduplicati per "first-segment").
  - `extract_diem_faculty_urls()`: scarica https://www.diem.unisa.it/
    dipartimento/personale, deriva le matricole dai link a rubrica.unisa.it,
    valida ogni profilo controllando la presenza di un link a
    docenti.unisa.it, e ritorna gli URL canonici "docenti.unisa.it/<id>/home".
  - `filter_docs(docs)`: rimuove docs con URL ".pdf", "/en/", "/zh/",
    "/print()", "?sitemap" e URL pre-2020 (regex anno).
  - `save_crawled_urls_to_json(docs, file)`,
    `save_crawled_pdfs_to_json(docs, file)`: dump JSON con
    {url, title} oppure {url, source_page, pages}.

Si interfaccia con: src/logger.py.
Librerie: json, re, time, urllib, requests, bs4 (BeautifulSoup),
langchain_community.

--------------------------------------------------------------------------------
5.2 parser.py
--------------------------------------------------------------------------------
Cosa fa:
  - `extract_html_metadata(html)`: estrae title, lang attribute, e prima
    data da `<meta>` o `<time datetime>`. Deve essere chiamato PRIMA di
    html_extractor (che distrugge il DOM).
  - `html_extractor(html)`: rimuove script/style/nav/footer/header/ecc.,
    prova selettori `main`/`article`/`#content`/...; restituisce testo
    pulito.
  - `should_keep_document(doc)` + `filter_recent_documents(docs)`: applica
    cutoff temporale 2020 leggendo metadata e prime TEMPORAL_SCAN_CHARS
    (2500) caratteri (con logica differenziata per pagine "time-sensitive":
    avvisi, news, bandi, seminari, eventi).
  - `load_pdfs_from_links(raw_docs, seen_urls)`: trova `<a href*.pdf>` nei
    documenti HTML grezzi e li scarica con PyPDFLoader; usa `seen_urls` per
    deduplicare; salta PDF in URL pre-2020.

Si interfaccia con: src/ingestion/crawler.py (is_pre_2020_url), src/logger.py.
Librerie: re, urllib, bs4, langchain_community (PyPDFLoader).

--------------------------------------------------------------------------------
5.3 enrichment.py
--------------------------------------------------------------------------------
Cosa fa: per ciascun documento, genera un "context header" di una frase
(<=15 parole effettive, controllo a 45 in fallback) che descrive il tipo
e l'oggetto del documento. Questo header viene poi prepended a ciascun
child chunk in fase di splitting.

Strategie:
  - `build_header_context(text, url)`: prepara evidenze (prime righe
    significative + passaggi keyword-based + estratti) e le passa a un
    prompt italiano.
  - `generate_context_header(text, url)`: POST a Ollama
    (llama3.2:3b di default, endpoint OLLAMA_ENDPOINT) con prompt RAG;
    se Ollama non risponde, disabilita la chiamata per il resto del run
    e usa `fallback_context_header` (euristica basata su keyword nell'URL
    + testo iniziale).
  - Cache in memoria (`_HEADER_CACHE`) chiave (url, primi 500 char).
  - `normalize_context_header(...)`: rimuove prefissi "Context:", clipping
    a 45 parole, fallback se output vuoto.
  - `add_context_headers(docs)`: scrive `doc.metadata["context_header"]`
    per ogni documento.

Si interfaccia con: src/ingestion/parser.py (clean_text), src/logger.py.
Endpoint esterno: Ollama HTTP API (default http://localhost:11434).
Librerie: os, re, requests, dotenv.

--------------------------------------------------------------------------------
5.4 database.py
--------------------------------------------------------------------------------
Cosa fa: Parent-Child Retriever indexing.

  - `ContextHeaderTextSplitter`: estende `RecursiveCharacterTextSplitter` e
    garantisce che ogni child chunk inizi con il context_header del proprio
    parent (evita duplicazione se gia' presente nel body).
  - `DocumentIndexer`:
      - Setup parent docstore (LocalFileStore su PARENT_STORE_DIR).
      - Inizializza Chroma con `hnsw:space=cosine`.
      - Split parent (PARENT_CHUNK_SIZE/OVERLAP).
      - Prepend header a ogni parent (per coerenza con i child).
      - Batch dei parent in base alla stima dei child generati
        (MAX_CHILD_CHUNKS_PER_BATCH=4000) e invoca
        `ParentDocumentRetriever.add_documents` per ciascun batch.
  - `index_documents(docs, embedding_model)`: entry point funzionale.

Si interfaccia con: config.py, src/logger.py.
Librerie: os, time, langchain_chroma, langchain_classic (retrievers,
storage), langchain_core, langchain_huggingface, langchain_text_splitters.

================================================================================
6. SCRIPT DI DEBUG - scripts/
================================================================================

scripts/check_vector_db.py
  Verifica che il context_header in metadata sia stato propagato in testa
  al page_content di ogni child chunk. Stampa il sample di 5 chunk e il
  tasso di propagazione. Richiede `chroma_diem/` popolato.

scripts/test_chunks.py
  Mostra statistiche e un esempio di:
    - child chunks letti da Chroma (dimensione, soglia, metadati);
    - parent chunks letti da LocalFileStore (parent_store).

scripts/test_retriever.py
  REPL interattiva: l'utente immette query + soglia + k, lo script
  istanzia un `ParentDocumentRetriever` (bi-encoder, no rerank) e stampa
  i documenti restituiti con URL, lunghezza, metadati e i primi 500
  caratteri.

Tutti gli script aggiungono PROJECT_ROOT a sys.path per riusare config.py
e src.brain. Non sono usati dall'applicazione a runtime.

================================================================================
7. EVALUATION FRAMEWORK - evaluation/
================================================================================

--------------------------------------------------------------------------------
7.1 evaluation/tester.py
--------------------------------------------------------------------------------
Runner ufficiale di valutazione.

Mappatura traccia -> metrica:
  Relevance       -> Ragas ResponseRelevancy
  Correctness     -> Ragas Faithfulness + AnswerCorrectness
                     (LLMContextPrecisionWithReference + LLMContextRecall)
  Coherence       -> Ragas AspectCritic (definito on-the-fly, LLM-as-judge)
  Robustness      -> Custom: scenari multi-turn (are_you_sure,
                     false_premise, leading_question, jailbreak,
                     role_override) giudicati da Ollama judge + double
                     check su marker di refusal.
  Scope Awareness -> Custom: rejection-phrase classifier (SCOPE_REJECTION
                     vs KNOWLEDGE_GAP markers) + LLM-judge. Distingue
                     strict pass (rifiuto esplicito) da soft pass
                     (knowledge-gap plea).

Flusso:
  1. Parse CLI (--lang, --limit, --categories, --judge-model,
     --skip-ragas, --ragas-metrics, --cache).
  2. Crea run dir `evaluation/results/<timestamp>_<lang>/` e setup logger.
  3. `load_brain` istanzia `DiemBrain` su Chroma esistente (fail-fast se
     l'indice manca).
  4. Per ogni categoria abilitata invoca:
       collect_rag_rows (in_scope, single-turn)
       collect_multi_turn_rag_rows (multi_turn)
       run_scope_awareness (out_of_scope)
       run_robustness (robustness, multi-turn adversarial)
     Tutte le invocazioni passano per `run_turn`, che usa la `TurnCache`
     opzionale.
  5. Ragas e' eseguito in due fasi:
       no-reference metrics su TUTTE le righe (incluse OOS/robustness)
       reference-requiring metrics solo su righe con `reference` non vuoto
     I DataFrame vengono mergiati su (user_input, response), il CSV
     viene flatten-ato per leggibilita'.
  6. `write_summary` produce `summary.md` con i 5 criteri della traccia.

Output per run:
  run.log, per_question.json, ragas_metrics.csv/.json,
  scope_awareness.json, robustness.json, summary.md.

Configurazione raccomandata per il judge: llama3.1:8b-instruct-q4_K_M
(piu robusto su JSON di small Ollama).

Si interfaccia con:
  brain.DiemBrain + embedding_model, config (Chroma path, modello chat),
  evaluation/cache.py.
Librerie: argparse, json, logging, pathlib, dataclasses, ragas,
langchain_chroma, langchain_ollama, langchain_core, warnings.

--------------------------------------------------------------------------------
7.2 evaluation/cache.py
--------------------------------------------------------------------------------
Cache su disco delle risposte chatbot, keyed per
(schema_version, chat_model, temperature, session_id, history pairs,
question). Tre modalita': off / use / refresh. Una cache hit reidrata la
history in-memory del chatbot con un (user, assistant) sintetico per
preservare la coerenza tra turni.

Storage: `evaluation/cache/<XX>/<sha256>.json` (sharding a 2 char).

NOTA: la cache NON e' invalidata automaticamente al cambio di
SYSTEM_PROMPT, parametri di retrieval o reindex Chroma. Vedi todo.txt
sezione "NOTE OPERATIVE".

--------------------------------------------------------------------------------
7.3 evaluation/dataset/
--------------------------------------------------------------------------------
golden_set_it.json + golden_set_en.json, schema:
  metadata: { version, language, description, evaluation_criteria }
  in_scope:    [ { id, persona, tags, question, reference } ]
  multi_turn:  [ { id, tag, turns: [ {question}, ... ], reference } ]
  out_of_scope:[ { id, question } ]
  robustness:  [ { id, tag, description, turns: [ {question}, ... ] } ]

I `reference` sono ground-truth approssimate (vedi todo.txt #0) e vanno
verificate a campione contro il sito DIEM live.

--------------------------------------------------------------------------------
7.4 evaluation/results/<run>/
--------------------------------------------------------------------------------
Generata automaticamente per ogni run; .gitkeep mantiene la dir vuota in
repo. Conserva tutti gli artefatti elencati al punto 7.1.

--------------------------------------------------------------------------------
7.5 evaluation/todo.txt
--------------------------------------------------------------------------------
Backlog operativo post-evaluation: bug confermati nel chatbot, run pendenti,
estensioni al golden set, idee per il tester. Etichette: [PRE-RUN], [BUG],
[ESTENSIONE]. Sezione finale "NOTE OPERATIVE" con istruzioni di gestione
della cache e modello judge.

================================================================================
8. LIBRERIE USATE (requirements.txt)
================================================================================

Core RAG (LangChain 1.x):
  langchain==1.2.15, langchain-core==1.3.2,
  langchain-community==0.4.1, langchain-text-splitters==1.1.2,
  langchain-huggingface==1.2.2, langchain-ollama==1.1.0.

Vector DB:
  chromadb==1.5.8, langchain-chroma==1.1.0.

UI:
  gradio==6.14.0.

Document processing:
  pypdf==6.10.2, beautifulsoup4==4.14.3, requests==2.32.5.

HuggingFace / ML:
  huggingface_hub==1.11.0, transformers==5.6.1,
  sentence-transformers==5.4.1, torch==2.11.0, numpy==2.4.3.

Evaluation:
  ragas==0.4.3 (con import legacy supportati).

Utility:
  python-dotenv==1.2.2.

Dipendenze esterne (NON in requirements):
  Ollama server in locale (default http://localhost:11434) per:
    - chat model (qwen2.5:7b)
    - context-header model in ingestion (llama3.2:3b)
    - judge model in evaluation (llama3.1:8b-instruct-q4_K_M)

================================================================================
9. FLUSSO COMPLETO DI ESECUZIONE
================================================================================

A) Setup
   python -m venv venv && venv\Scripts\activate
   pip install -r requirements.txt
   # avviare Ollama e pullare i modelli necessari
   ollama pull qwen2.5:7b
   ollama pull llama3.2:3b
   ollama pull llama3.1:8b-instruct-q4_K_M    # solo per evaluation

B) Ingestion (una-tantum o on-demand)
   python main_ingestion.py --crawl-only      # opzionale, dump JSON
   python main_ingestion.py --full            # crawl + filter + embed + index

C) Avvio UI
   python app.py                              # carica chroma_diem/
   python app.py --reindex                    # forza una full ingestion

D) Debug
   python scripts/check_vector_db.py
   python scripts/test_chunks.py
   python scripts/test_retriever.py

E) Evaluation
   python evaluation/tester.py --lang it
   python evaluation/tester.py --lang it --limit 3 --ragas-metrics coherence
   python evaluation/tester.py --lang it --cache use

================================================================================
10. VARIABILI D'AMBIENTE (.env)
================================================================================
Tutte opzionali; ognuna ha un default in config.py. Le piu rilevanti:
  LOG_LEVEL, MAX_LOG_SIZE_MB, LOG_BACKUP_COUNT
  OLLAMA_CHAT_MODEL, LLM_TEMPERATURE
  EMBEDDING_MODEL_NAME, CROSS_ENCODER_MODEL_NAME
  BI_ENCODER_K, CROSS_ENCODER_K, RETRIEVER_SCORE_THRESHOLD
  OLLAMA_ENDPOINT (usato da enrichment.py)

================================================================================
