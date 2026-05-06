import os
import sys
import time
import re
os.environ.setdefault("PYTHONUNBUFFERED", "1")
from urllib.parse import urljoin, urlparse, urldefrag
from dotenv import load_dotenv
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_community.document_loaders import PyPDFLoader, RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from bs4 import BeautifulSoup

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_DIR = "chroma_diem"
COLLECTION  = "diem_knowledge"
PARENT_STORE_DIR = os.path.join(CHROMA_DIR, "parent_store")
HTML_LINK_REGEX = r"""<a\s+(?:[^>]*?\s+)?href=["']([^"']*)["']"""
MAX_CHILD_CHUNKS_PER_BATCH = 4000
YEAR_CUTOFF = 2020
TEMPORAL_SCAN_CHARS = 2500
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
_HEADER_CACHE = {}
_OLLAMA_DISABLED = False

METADATA_DATE_KEYS = (
    "date", "created", "creation_date", "creationdate", "moddate",
    "modified", "last_modified", "published", "publish_date", "updated",
)

EXCLUDE_DIRS = [
    "/rescue/css/", "/rescue/js/", "/rescue/assets/",
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".ico", ".woff", ".woff2", ".ttf", ".eot",
    "/idp/", "/password-recovery", "/login",
]


# ─────────────────────────────────────────────────────────────────────────────
# HTML Extractor
# Converts HTML to clean text and removes noise: nav, footer, scripts, ads.
# ─────────────────────────────────────────────────────────────────────────────
def html_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "aside", "iframe"]):
        tag.decompose()
    for selector in ["main", "article", "#content", ".content",
                     "#main", ".main-content", ".entry-content"]:
        content = soup.select_one(selector)
        if content:
            return clean_text(content.get_text("\n", strip=True))
    body = soup.find("body")
    source = body if body else soup
    return clean_text(source.get_text("\n", strip=True))


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_years_from_text(text: str) -> list[int]:
    if not text:
        return []
    return [int(year) for year in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]


def extract_years_from_metadata(metadata: dict) -> list[int]:
    years = []
    for key, value in metadata.items():
        key_name = key.lower().strip("/")
        if value is None or key_name not in METADATA_DATE_KEYS:
            continue

        years.extend(extract_years_from_text(str(value)))

    return years


def should_keep_document(doc) -> tuple[bool, str]:
    source = doc.metadata.get("source", "")
    scan_text = doc.page_content[:TEMPORAL_SCAN_CHARS]
    lowered = f"{source}\n{scan_text}".lower()
    is_pdf = ".pdf" in urlparse(source).path.lower()
    is_time_sensitive = any(
        marker in lowered
        for marker in ("avvisi", "avviso", "news", "bando", "bandi", "seminari", "eventi")
    )

    metadata_or_url_years = (
        extract_years_from_metadata(doc.metadata)
        + extract_years_from_text(source)
    )
    if metadata_or_url_years:
        newest = max(metadata_or_url_years)
        if newest < YEAR_CUTOFF:
            return False, f"old year {newest}"
        return True, f"year {newest}"

    text_years = extract_years_from_text(scan_text)
    if (is_pdf or is_time_sensitive) and text_years:
        newest = max(text_years)
        if newest < YEAR_CUTOFF:
            return False, f"old text year {newest}"
        return True, f"text year {newest}"

    return True, "no explicit date"


def filter_recent_documents(docs: list) -> list:
    kept = []
    dropped = 0
    for doc in docs:
        keep, reason = should_keep_document(doc)
        if keep:
            doc.metadata["temporal_filter"] = reason
            kept.append(doc)
        else:
            dropped += 1
            source = doc.metadata.get("source", "unknown source")
            print(f"  SKIP old document ({reason}): {source}")

    print(
        f"  -> Temporal filter kept {len(kept)}/{len(docs)} documents "
        f"(cutoff year: {YEAR_CUTOFF}, dropped: {dropped})"
    )
    return kept


def fallback_context_header(text: str, url: str) -> str:
    combined = f"{url}\n{text[:700]}".lower()
    if "docenti.unisa.it" in combined or "professore" in combined or "docente" in combined:
        return "Profile and contact info of a DIEM professor."
    if "corsi.unisa.it" in combined or "corso di laurea" in combined or "insegnamento" in combined:
        return "Syllabus and information for a DIEM course."
    if "ufficio" in combined or "segreteria" in combined or "servizio" in combined:
        return "Physical location, office hours, and contact points for DIEM."
    if "avvisi" in combined or "avviso" in combined or "news" in combined:
        return "Official notice regarding DIEM."
    return "General information page about DIEM."


def generate_context_header(text: str, url: str) -> str:
    global _OLLAMA_DISABLED
    cache_key = (url, text[:500])
    if cache_key in _HEADER_CACHE:
        return _HEADER_CACHE[cache_key]

    if _OLLAMA_DISABLED:
        header = fallback_context_header(text, url)
        _HEADER_CACHE[cache_key] = header
        return header

    prompt = f"""
ROLE: Assistant for Academic Data Indexing.
TASK: Analyze the provided text from the University of Salerno, DIEM department.
OUTPUT: A single concise sentence, max 15 words, identifying the subject and context.

CONTEXT RULES:
If it is a person: "Profile and contact info of Prof. [Name] (DIEM)."
If it is a course: "Syllabus and info for the course [Course Name] (DIEM)."
If it is a location or office page: "Physical location, office hours, and contact points for DIEM."
If it is a notice: "Official notice regarding [Subject] at DIEM."
If the subject is unclear: "General information page about DIEM."

RULES:
Do not invent names, course titles, or subjects.
Use only information explicitly present in the text or URL.
Return only the sentence.
Do not add explanations, bullets, quotes, or labels.

TEXT:
{text[:1200]}

URL:
{url}

RESPONSE:
""".strip()


    try:
        response = requests.post(
            OLLAMA_ENDPOINT,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 40},
            },
            timeout=20,
        )
        response.raise_for_status()
        header = response.json().get("response", "").strip().splitlines()[0]
        header = clean_text(header.strip("\"' "))
        words = header.split()
        if not header or len(words) > 18:
            header = fallback_context_header(text, url)
    except Exception as e:
        _OLLAMA_DISABLED = True
        print(f"  WARNING: Ollama unavailable, using heuristic context headers: {e}")
        header = fallback_context_header(text, url)

    if not header.lower().startswith("context:"):
        header = f"Context: {header}"

    # Enforce the requested compactness even if the local model is verbose.
    words = header.split()
    if len(words) > 15:
        header = " ".join(words[:15]).rstrip(".,;:") + "."

    _HEADER_CACHE[cache_key] = header
    return header


def add_context_headers(docs: list) -> None:
    print("\nAdding contextual headers with Ollama...")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "")
        header = generate_context_header(doc.page_content, source)
        doc.metadata["context_header"] = header
        doc.page_content = f"{header}\n\n{doc.page_content}"
        if i % 100 == 0 or i == len(docs):
            print(f"  -> {i}/{len(docs)} contextual headers added", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: extract links to docenti/corsi from raw HTML docs
# ─────────────────────────────────────────────────────────────────────────────
def extract_external_links(raw_docs: list) -> dict:
    targets = {"docenti.unisa.it": set(), "corsi.unisa.it": set()}
    for doc in raw_docs:
        try:
            soup = BeautifulSoup(doc.page_content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href.startswith("http"):
                    continue
                for domain in targets:
                    if domain in href:
                        clean = href.split("?")[0].split("#")[0].rstrip("/")
                        targets[domain].add(clean)
        except Exception:
            pass
    return targets


def load_pdfs_from_links(raw_docs: list, seen_urls: set | None = None) -> list:
    """Load PDF documents linked from already crawled HTML pages."""
    seen_urls = seen_urls if seen_urls is not None else set()
    pdf_docs = []

    for doc in raw_docs:
        page_url = doc.metadata.get("source", "")
        try:
            soup = BeautifulSoup(doc.page_content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                clean_href = href.split("?")[0].split("#")[0]
                if not clean_href.lower().endswith(".pdf"):
                    continue

                pdf_url, _ = urldefrag(urljoin(page_url, href))
                if pdf_url in seen_urls:
                    continue
                seen_urls.add(pdf_url)

                try:
                    docs = PyPDFLoader(pdf_url).load()
                    for pdf_doc in docs:
                        pdf_doc.metadata.setdefault("source", pdf_url)
                        pdf_doc.metadata.setdefault("source_page", page_url)
                    pdf_docs.extend(docs)
                    print(f"  PDF loaded: {pdf_url} ({len(docs)} pages)")
                except Exception as e:
                    print(f"  WARNING: skipped PDF {pdf_url}: {e}")
        except Exception as e:
            print(f"  WARNING: could not inspect PDF links in {page_url}: {e}")

    return pdf_docs


def get_section_base(url: str) -> str:
    """Return the first-segment root of a URL path.

    https://docenti.unisa.it/003145/home  ->  https://docenti.unisa.it/003145/
    https://corsi.unisa.it/ing-inf/home   ->  https://corsi.unisa.it/ing-inf/
    """
    parsed    = urlparse(url)
    parts     = parsed.path.strip("/").split("/")
    base_path = f"/{parts[0]}/" if parts and parts[0] else "/"
    return f"{parsed.scheme}://{parsed.netloc}{base_path}"


# ─────────────────────────────────────────────────────────────────────────────
# Crawl helper
# ─────────────────────────────────────────────────────────────────────────────
def crawl(start_url: str, base_url: str, max_depth: int = 2) -> list:
    try:
        loader = RecursiveUrlLoader(
            start_url,
            base_url=base_url,
            max_depth=max_depth,
            prevent_outside=True,
            timeout=15,
            check_response_status=True,
            exclude_dirs=EXCLUDE_DIRS,
            link_regex=HTML_LINK_REGEX,
        )
        return loader.load()
    except Exception as e:
        print(f"  FAILED {start_url}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: crawl -> load PDFs -> parent-child index
# ─────────────────────────────────────────────────────────────────────────────
def build_index(embedding_model: HuggingFaceEmbeddings) -> Chroma:

    # Prima esecuzione con la nuova struttura Parent-Child: elimina la cartella
    # "chroma_diem" esistente per evitare conflitti con il vecchio indice.

    # ── PHASE 1: Load ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 – Loading web pages")
    print("=" * 60)

    all_docs = []
    seen_pdf_urls = set()

    # 1a. Crawl diem.unisa.it WITHOUT extractor to keep raw HTML for link extraction
    print("\n[1/3] Crawling www.diem.unisa.it ...")
    raw_loader = RecursiveUrlLoader(
        "https://www.diem.unisa.it/",
        base_url="https://www.diem.unisa.it/",
        max_depth=3,
        prevent_outside=True,
        timeout=15,
        check_response_status=True,
        exclude_dirs=EXCLUDE_DIRS,
        link_regex=HTML_LINK_REGEX,
    )
    raw_diem = raw_loader.load()
    print(f"  -> {len(raw_diem)} pages found")

    # Extract links before cleaning
    external     = extract_external_links(raw_diem)
    docenti_urls = sorted(external["docenti.unisa.it"])
    corsi_urls   = sorted(external["corsi.unisa.it"])
    print(f"  -> {len(docenti_urls)} docenti links, {len(corsi_urls)} corsi links")
    print("\nDocenti URLs found:")
    for url in docenti_urls:
        print(f"  {url}")

    diem_pdf_docs = load_pdfs_from_links(raw_diem, seen_pdf_urls)

    # Clean diem docs in-place
    for doc in raw_diem:
        doc.page_content = html_extractor(doc.page_content)
    all_docs.extend(raw_diem)
    all_docs.extend(diem_pdf_docs)

    # 1b. Crawl docenti.unisa.it
    total_docenti = len(docenti_urls)
    print(f"\n[2/3] Crawling docenti.unisa.it ({total_docenti} faculty pages) ...")
    for i, url in enumerate(docenti_urls, 1):
        base = get_section_base(url)
        docs = crawl(url, base_url=base, max_depth=2)
        pdf_docs = load_pdfs_from_links(docs, seen_pdf_urls)
        for doc in docs:
            doc.page_content = html_extractor(doc.page_content)
        all_docs.extend(docs)
        all_docs.extend(pdf_docs)
        print(f"  [{i:02d}/{total_docenti}] {url}  ({len(docs)} sub-pages)")

    # 1c. Crawl corsi.unisa.it
    total_corsi = len(corsi_urls)
    print(f"\n[3/3] Crawling corsi.unisa.it ({total_corsi} course pages) ...")
    for i, url in enumerate(corsi_urls, 1):
        base = get_section_base(url)
        docs = crawl(url, base_url=base, max_depth=2)
        pdf_docs = load_pdfs_from_links(docs, seen_pdf_urls)
        for doc in docs:
            doc.page_content = html_extractor(doc.page_content)
        all_docs.extend(docs)
        all_docs.extend(pdf_docs)
        print(f"  [{i:02d}/{total_corsi}] {url}  ({len(docs)} sub-pages)")

    print(f"\nTotal documents loaded before temporal filtering: {len(all_docs)}")
    all_docs = filter_recent_documents(all_docs)
    add_context_headers(all_docs)
    print(f"\nTotal documents loaded: {len(all_docs)}")

    # ── PHASE 2: Parent-Child splitting ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 – Parent-Child splitters")
    print("=" * 60)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50
    )
    parent_docs = parent_splitter.split_documents(all_docs)
    print(f"  -> {len(parent_docs)} parent documents from {len(all_docs)} sources")

    # ── PHASE 3: Embed & Index ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 – Embedding child chunks and indexing parents")
    print("=" * 60)

    child_vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )

    os.makedirs(PARENT_STORE_DIR, exist_ok=True)
    parent_store = LocalFileStore(PARENT_STORE_DIR)
    parent_docstore = create_kv_docstore(parent_store)

    retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=parent_docstore,
        # I parent sono gia' stati creati in PHASE 2: indicizzarli direttamente
        # evita batch troppo grandi per Chroma quando una sorgente e' molto lunga.
        child_splitter=child_splitter,
    )

    indexed_parent_docs = 0
    start_time = time.time()
    batch = []
    batch_child_chunks = 0

    def index_batch(batch: list, batch_child_chunks: int) -> None:
        nonlocal indexed_parent_docs
        if not batch:
            return

        retriever.add_documents(batch)

        indexed_parent_docs += len(batch)
        elapsed = time.time() - start_time

        try:
            child_chunks = child_vectorstore._collection.count()
            child_info = f", child chunks in Chroma: {child_chunks}"
        except Exception:
            child_info = ""

        print(
            f"  -> {indexed_parent_docs}/{len(parent_docs)} parent docs indexed "
            f"({indexed_parent_docs / len(parent_docs):.1%}); "
            f", batch child chunks: {batch_child_chunks}"
            f"{child_info}; elapsed: {elapsed / 60:.1f} min",
            flush=True,
        )

    for parent_doc in parent_docs:
        doc_child_chunks = len(child_splitter.split_documents([parent_doc]))

        if (
            batch
            and batch_child_chunks + doc_child_chunks > MAX_CHILD_CHUNKS_PER_BATCH
        ):
            index_batch(batch, batch_child_chunks)
            batch = []
            batch_child_chunks = 0

        batch.append(parent_doc)
        batch_child_chunks += doc_child_chunks

    index_batch(batch, batch_child_chunks)

    print(f"\nIndexing complete. Parent documents indexed: {len(parent_docs)}")
    return child_vectorstore


if __name__ == "__main__":
    from brain import embedding_model
    build_index(embedding_model)
