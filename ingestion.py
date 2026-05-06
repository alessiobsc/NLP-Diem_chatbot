import os
import sys
os.environ.setdefault("PYTHONUNBUFFERED", "1")
from urllib.parse import urlparse
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from bs4 import BeautifulSoup

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_DIR = "chroma_diem"
COLLECTION  = "diem_knowledge"

EXCLUDE_DIRS = [
    "/rescue/css/", "/rescue/js/", "/rescue/assets/",
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".ico", ".woff", ".woff2", ".ttf", ".eot",
    "/idp/", "/password-recovery", "/login",
]


# ─────────────────────────────────────────────────────────────────────────────
# HTML Extractor
# Keeps heading tags (h1/h2/h3) so HTMLSectionSplitter can split on them.
# Removes noise: nav, footer, scripts, ads.
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
            return str(content)
    body = soup.find("body")
    return str(body) if body else html


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
            extractor=html_extractor,
            exclude_dirs=EXCLUDE_DIRS,
        )
        return loader.load()
    except Exception as e:
        print(f"  FAILED {start_url}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: crawl -> chunk -> embed -> store
# ─────────────────────────────────────────────────────────────────────────────
def build_index(embedding_model: HuggingFaceEmbeddings) -> Chroma:

    # ── PHASE 1: Load ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 – Loading web pages")
    print("=" * 60)

    all_docs = []

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
    )
    raw_diem = raw_loader.load()
    print(f"  -> {len(raw_diem)} pages found")

    # Extract links before cleaning
    external     = extract_external_links(raw_diem)
    docenti_urls = list(external["docenti.unisa.it"])
    corsi_urls   = list(external["corsi.unisa.it"])
    print(f"  -> {len(docenti_urls)} docenti links, {len(corsi_urls)} corsi links")

    # Clean diem docs in-place
    for doc in raw_diem:
        doc.page_content = html_extractor(doc.page_content)
    all_docs.extend(raw_diem)

    # 1b. Crawl docenti.unisa.it  (cap at 50 for prototype)
    cap_docenti = min(50, len(docenti_urls))
    print(f"\n[2/3] Crawling docenti.unisa.it ({cap_docenti} faculty pages) ...")
    for i, url in enumerate(docenti_urls[:cap_docenti], 1):
        base = get_section_base(url)
        docs = crawl(url, base_url=base, max_depth=2)
        all_docs.extend(docs)
        print(f"  [{i:02d}/{cap_docenti}] {url}  ({len(docs)} sub-pages)")

    # 1c. Crawl corsi.unisa.it  (cap at 30 for prototype)
    cap_corsi = min(30, len(corsi_urls))
    print(f"\n[3/3] Crawling corsi.unisa.it ({cap_corsi} course pages) ...")
    for i, url in enumerate(corsi_urls[:cap_corsi], 1):
        base = get_section_base(url)
        docs = crawl(url, base_url=base, max_depth=2)
        all_docs.extend(docs)
        print(f"  [{i:02d}/{cap_corsi}] {url}  ({len(docs)} sub-pages)")

    print(f"\nTotal documents loaded: {len(all_docs)}")

    # ── PHASE 2: Chunk ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 – Chunking")
    print("=" * 60)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    )
    chunks = char_splitter.split_documents(all_docs)
    print(f"  -> {len(chunks)} chunks from {len(all_docs)} documents")

    # ── PHASE 3: Embed & Index ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 – Embedding and indexing")
    print("=" * 60)

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        vectorstore.add_documents(chunks[i : i + batch_size])
        print(f"  -> {min(i + batch_size, len(chunks))}/{len(chunks)} chunks indexed", flush=True)

    print("\nIndexing complete.")
    return vectorstore


if __name__ == "__main__":
    from brain import embedding_model
    build_index(embedding_model)
