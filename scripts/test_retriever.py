"""
Script to test the retrieval of documents from Qdrant using QdrantRAG.
Allows the user to input a query and see the retrieved parent documents.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import QDRANT_STORAGE_DIR
from src.rag_hybrid import QdrantRAG

def get_qdrant_rag() -> QdrantRAG:
    """
    Initializes and returns the QdrantRAG system.
    """
    return QdrantRAG(in_memory=False)


def main() -> None:
    """Main execution block for the interactive retrieval test."""
    print("=== DIEM Chatbot - Dense Parent-Child Retrieval Test ===")
    
    try:
        rag = get_qdrant_rag()
        print(f"Connected to Qdrant at {QDRANT_STORAGE_DIR}")
    except Exception as e:
        print(f"ERROR: Could not initialize QdrantRAG. ({e})")
        return

    while True:
        try:
            query = input("\nInserisci la query di ricerca (o 'q' per uscire): ").strip()
            if query.lower() in ['q', 'quit', 'exit']:
                break
                
            if not query:
                continue
            
            k_input = input("Inserisci il numero massimo di risultati (k) [default 5]: ").strip()
            k = int(k_input) if k_input else 5
            
            print(f"\nRicerca in corso per: '{query}' (k: {k})...")
            
            parent_docs = rag.search_and_retrieve_parents(query, limit=k)
            
            if not parent_docs:
                print("Nessun documento trovato che soddisfi i criteri.")
                continue
                
            print(f"\nTrovati {len(parent_docs)} documenti parent.")
            print("-" * 50)
            
            for i, doc in enumerate(parent_docs, 1):
                source = doc.metadata.get('source', 'Sconosciuto')
                header = doc.metadata.get('context_header', '')

                print(f"\n--- Documento Parent {i} ---")
                print(f"URL: {source}")
                if header:
                    print(f"Header: {header[:100]}...")
                print(f"Lunghezza contenuto: {len(doc.page_content)} caratteri")
                print("\nContenuto (primi 500 caratteri):")
                print(doc.page_content[:500] + "...")
                print("-" * 50)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nErrore durante la ricerca: {e}")
            
    print("\nUscita.")


if __name__ == "__main__":
    main()
