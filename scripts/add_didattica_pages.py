"""
Crawl mirato incrementale: aggiunge alla collection Chroma esistente le pagine
"didattica?anno=YYYY&id=..." (testi di riferimento/consultazione per corso) di
tutti i docenti DIEM validati. Non tocca il resto della collection.

Gap riscontrato: il crawl full-pipeline arriva a queste pagine (sono a un hop
dalla pagina /didattica base) ma per la maggior parte dei docenti non risultano
in crawled_urls.json — probabile dedup/visited-cache troppo aggressiva su
query string diverse dallo stesso path. Non investigato a fondo: qui si
aggira il problema con un fetch mirato invece di rifare il crawl completo.
"""
import datetime
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME
from main_ingestion import apply_html_metadata_and_filter, normalized_content_hash
from src.encoders.embedding_init import build_embedding_model
from src.ingestion.crawler import crawl, extract_diem_faculty_urls, get_section_base
from src.ingestion.database import index_documents
from src.ingestion.parser import filter_low_quality_documents, filter_recent_documents
from src.utils.logger import get_logger

logger = get_logger(__name__)

CURRENT_YEAR = datetime.datetime.now().year
ID_LINK_RE = re.compile(rf"didattica\?anno={CURRENT_YEAR}&id=(\d+)")


def dedupe_by_content(docs: list) -> list:
    """Content-hash dedup across docenti (co-docenza: stesso corso, stesso
    testo, URL diverse per matricola). Sicuro solo qui perche' il batch e'
    omogeneo (sole pagine corso) - dedupe_docs_by_source_alias_and_content in
    main_ingestion.py e' deliberatamente piu' cauta e non collasserebbe questo
    caso, perche' richiede anche l'alias URL uguale (qui non lo e').

    La prima riga del testo estratto e' l'h1 "{Docente} | {Corso}" (vedi
    header_heuristic.py) ed e' l'unica parte che varia tra co-docenti sulla
    stessa pagina corso - va esclusa dall'hash, altrimenti il contenuto
    identico sottostante non viene mai deduplicato."""
    seen: set[str] = set()
    unique = []
    for doc in docs:
        body = doc.page_content.split("\n", 1)[-1]
        content_hash = normalized_content_hash(body)
        if content_hash and content_hash in seen:
            continue
        if content_hash:
            seen.add(content_hash)
        unique.append(doc)
    removed = len(docs) - len(unique)
    if removed:
        logger.info(f"  -> Content dedup (co-docenza): rimossi {removed} duplicati")
    return unique


def collect_course_urls() -> set[str]:
    """Fetch ogni pagina /didattica base dei docenti, estrae i link ai corsi
    dell'anno corrente.

    L'id numerico NON e' univoco tra docenti diversi (stesso id puo' comparire
    per corsi in co-docenza sotto matricole diverse -> stesso corso, contenuto
    identico). Qui si raccolgono tutte le URL complete (docente+id) senza
    scartare nulla; la deduplicazione dei contenuti realmente identici avviene
    dopo il fetch, per content-hash, in dedupe_by_content.
    """
    docenti_urls = sorted(extract_diem_faculty_urls())
    logger.info(f"{len(docenti_urls)} docenti DIEM validati")

    course_urls: set[str] = set()
    for i, url in enumerate(docenti_urls, 1):
        base = get_section_base(url)
        didattica_url = f"{base}didattica"
        docs = list(crawl(didattica_url, base_url=didattica_url, max_depth=1))
        if not docs:
            logger.warning(f"[{i}/{len(docenti_urls)}] nessuna pagina per {didattica_url}")
            continue

        found_ids = set(ID_LINK_RE.findall(docs[0].page_content))
        for cid in found_ids:
            course_urls.add(f"{base}didattica?anno={CURRENT_YEAR}&id={cid}")

        logger.info(f"[{i}/{len(docenti_urls)}] {didattica_url} -> {len(found_ids)} corsi")

    return course_urls


def purge_existing_chunks(course_urls: set[str]) -> None:
    """Delete any previously-indexed child chunks for these source URLs before
    re-adding them.

    Needed because ParentDocumentRetriever.add_documents (langchain_classic)
    only accepts explicit `ids` for the PARENT docs; the CHILD chunks it writes
    to the Chroma vectorstore always get a fresh uuid4(), never a deterministic
    id. Re-running index_documents() on the same URLs therefore duplicates
    child chunks instead of overwriting them — this purge is what makes
    re-running this script idempotent.
    """
    if not course_urls:
        return
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    before = collection.count()
    collection.delete(where={"source": {"$in": sorted(course_urls)}})
    after = collection.count()
    logger.info(f"Purge pre-esistenti per queste URL: {before - after} chunk rimossi (before={before}, after={after})")


def fetch_course_pages(course_urls: set[str]) -> list:
    raw_docs = []
    items = sorted(course_urls)
    for i, url in enumerate(items, 1):
        raw_docs.extend(crawl(url, base_url=url, max_depth=1))
        if i % 25 == 0 or i == len(items):
            logger.info(f"  fetch pagine corso: {i}/{len(items)}")
    return raw_docs


def main() -> None:
    course_urls = collect_course_urls()
    logger.info(f"Pagine corso da scaricare (anno {CURRENT_YEAR}): {len(course_urls)}")

    purge_existing_chunks(course_urls)

    raw_docs = fetch_course_pages(course_urls)
    logger.info(f"Pagine corso scaricate: {len(raw_docs)}")

    docs = apply_html_metadata_and_filter(raw_docs)
    docs = dedupe_by_content(docs)
    docs = filter_recent_documents(docs)
    docs = filter_low_quality_documents(docs)
    logger.info(f"Documenti finali da indicizzare: {len(docs)}")

    if not docs:
        logger.info("Nulla da indicizzare, esco.")
        return

    embedding_model = build_embedding_model()
    index_documents(docs, embedding_model)
    logger.info("Aggiunta incrementale completata: collection esistente estesa, non ricostruita.")


if __name__ == "__main__":
    main()