import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.ingestion.crawl_state import CrawlStateManager
from src.ingestion.crawler import crawl_html_sitemap, filter_docs as crawler_filter_docs, extract_diem_faculty_urls, \
    crawl as crawl_single_url
from src.ingestion.parser import (
    html_extractor_for_source,
    filter_recent_documents,
    filter_low_quality_documents,
    load_pdfs_from_links
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_update():
    """
    Esegue l'aggiornamento completo e dinamico della collection ChromaDB.
    1. Fa uno snapshot degli URL esistenti.
    2. Esegue un crawl partendo da URL "seed" per trovare documenti nuovi o modificati.
    3. Processa, pulisce e filtra i documenti (HTML e PDF).
    4. Li divide in chunk e li indicizza (upsert) in ChromaDB.
    5. Rimuove dalla collection i documenti che non sono più raggiungibili.
    """
    logger.info("--- Inizio aggiornamento dinamico della collection ---")
    collection_name = "diem_chatbot_collection"
    
    # URL di partenza per il crawler
    diem_seed_url = "https://www.diem.unisa.it/"
    corsi_seed_url = "https://corsi.unisa.it/"

    # Inizializza il text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )

    with CrawlStateManager() as crawl_state:
        # --- 1. Inizializzazione e Snapshot ---
        try:
            chroma_client = chromadb.HttpClient(host='localhost', port=8000)
            collection = chroma_client.get_or_create_collection(collection_name)
            logger.info(f"Connesso a ChromaDB. Collection '{collection_name}' pronta.")
        except Exception as e:
            logger.error(f"Impossibile connettersi a ChromaDB: {e}")
            return

        previously_crawled_urls = set(crawl_state.get_all_urls())
        logger.info(f"Stato precedente: {len(previously_crawled_urls)} URL indicizzati.")

        # --- 2. Crawling ---
        # Il crawler usa internamente lo stato per essere efficiente
        logger.info("Avvio del crawling dal sitemap DIEM...")
        raw_docs_diem = crawl_html_sitemap(diem_seed_url, max_depth=2)

        logger.info("Avvio del crawling dal sitemap Corsi...")
        raw_docs_corsi = crawl_html_sitemap(corsi_seed_url, max_depth=1)

        # Aggiungiamo gli URL dei docenti
        logger.info("Recupero degli URL dei docenti DIEM...")
        faculty_urls = extract_diem_faculty_urls()
        raw_docs_faculty = []
        for url in faculty_urls:
            # Usiamo crawl_single_url che rispetta il CrawlStateManager
            raw_docs_faculty.extend(crawl_single_url(url, base_url=url, max_depth=1))

        # Uniamo tutti i documenti grezzi
        all_raw_docs = list(raw_docs_diem) + list(raw_docs_corsi) + list(raw_docs_faculty)
        logger.info(f"Crawl iniziale completato. Trovati {len(all_raw_docs)} documenti HTML grezzi.")

        # --- 3. Parsing, Filtro e Arricchimento ---
        # Primo filtro e deduplica
        base_docs = crawler_filter_docs(all_raw_docs)
        logger.info(f"Documenti dopo il primo filtro e deduplica: {len(base_docs)}")

        # Estrazione dei PDF linkati
        seen_urls_for_pdfs = {doc.metadata.get("source") for doc in base_docs}
        pdf_docs = load_pdfs_from_links(base_docs, seen_urls=seen_urls_for_pdfs)
        logger.info(f"Trovati e caricati {len(pdf_docs)} documenti PDF.")

        # Uniamo HTML e PDF
        all_docs = base_docs + pdf_docs
        
        # Estrazione del testo pulito
        logger.info("Estrazione del testo pulito dai documenti...")
        for doc in all_docs:
            if ".pdf" not in doc.metadata.get("source", "").lower():
                doc.page_content = html_extractor_for_source(doc.page_content, doc.metadata.get("source", ""))

        # Filtri di qualità e temporali
        quality_docs = filter_low_quality_documents(all_docs)
        final_docs = filter_recent_documents(quality_docs)
        
        logger.info(f"Pipeline di pulizia completata. Documenti finali da indicizzare: {len(final_docs)}")

        # --- 4. Chunking e Indicizzazione ---
        currently_reachable_urls = set()
        
        if not final_docs:
            logger.info("Nessun documento da indicizzare in questa sessione.")
        else:
            logger.info("Inizio chunking e indicizzazione...")
            # Splitting dei documenti in chunk
            chunks = text_splitter.split_documents(final_docs)
            logger.info(f"Creati {len(chunks)} chunk totali.")

            # Preparazione per l'upsert
            ids = [f"{chunk.metadata.get('source')}-{i}" for i, chunk in enumerate(chunks)]
            contents = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Aggiungiamo gli URL processati al set
            for meta in metadatas:
                if 'source' in meta:
                    currently_reachable_urls.add(meta['source'])

            # Eseguiamo l'upsert in batch (Chroma gestisce il batching internamente)
            try:
                collection.upsert(ids=ids, documents=contents, metadatas=metadatas)
                logger.info(f"Upsert completato per {len(chunks)} chunk.")
            except Exception as e:
                logger.error(f"Errore durante l'operazione di upsert su ChromaDB: {e}")

        # --- 5. Riconciliazione e Pulizia ---
        urls_to_delete = previously_crawled_urls - currently_reachable_urls
        if not urls_to_delete:
            logger.info("Nessun URL obsoleto da eliminare.")
        else:
            logger.warning(f"Trovati {len(urls_to_delete)} URL da eliminare: {urls_to_delete}")
            for url in urls_to_delete:
                try:
                    # Elimina da ChromaDB usando il filtro sui metadati
                    collection.delete(where={"source": url})
                    # Rimuovi dallo stato
                    crawl_state.remove_url(url)
                    logger.info(f"Eliminati i dati per l'URL obsoleto: {url}")
                except Exception as e:
                    logger.error(f"Errore durante l'eliminazione di {url}: {e}")

    logger.info("--- Aggiornamento dinamico completato ---")


if __name__ == "__main__":
    run_update()
