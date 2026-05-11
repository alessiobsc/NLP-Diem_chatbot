"""
Analisi dei chunk salvati nel Chroma DB e nel parent store.
Verifica la corretta separazione in chunks con i relativi metadati.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHROMA_DIR, COLLECTION_NAME, PARENT_STORE_DIR,
    CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE
)
from src.brain import embedding_model


def main() -> None:
    if not CHROMA_DIR.exists():
        print("ERROR: chroma_diem/ not found. Run --full first.")
        return

    from langchain_chroma import Chroma
    from langchain_classic.storage import LocalFileStore, create_kv_docstore

    print("Connecting to Chroma...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("\nConnecting to Parent Store...")
    fs = LocalFileStore(str(PARENT_STORE_DIR))
    parent_store = create_kv_docstore(fs)

    print("\n" + "="*50)
    print("CHILD CHUNKS (Chroma vector store)")
    print("="*50)

    # Check what Chroma returns
    try:
        data = vectorstore.get()
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])

        print(f"Total child chunks found: {len(docs)}")

        if docs:
            print("\nEsempio di Child Chunk:")
            print(f"Dimensione (caratteri): {len(docs[0])}")
            print(f"Soglia impostata: {CHILD_CHUNK_SIZE}")
            print("Contenuto (primi 200 caratteri):")
            print(docs[0][:200] + "...")
            print("Metadati:")
            print(metas[0])
            print("-" * 30)

    except Exception as e:
        print(f"Errore nella lettura da Chroma: {e}")

    print("\n" + "="*50)
    print("PARENT CHUNKS (LocalFileStore)")
    print("="*50)

    try:
        # parent_store yields (key, Document)
        items = list(parent_store.yield_keys())
        print(f"Total parent chunks found: {len(items)}")

        if items:
            first_key = items[0]
            first_doc = parent_store.mget([first_key])[0]

            print("\nEsempio di Parent Chunk:")
            print(f"ID (doc_id): {first_key}")
            if first_doc:
                print(f"Dimensione (caratteri): {len(first_doc.page_content)}")
                print(f"Soglia impostata: {PARENT_CHUNK_SIZE}")
                print("Contenuto (primi 200 caratteri):")
                print(first_doc.page_content[:200] + "...")
                print("Metadati:")
                print(first_doc.metadata)
            else:
                print("Documento non trovato per questa chiave.")
            print("-" * 30)

    except Exception as e:
        print(f"Errore nella lettura del parent store: {e}")


if __name__ == "__main__":
    main()
