"""
Script to test the retrieval of documents from Chroma DB.
Allows the user to input a query and a score threshold, displaying the retrieved documents.
"""
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers.multi_vector import SearchType
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.brain import embedding_model
from config import CHROMA_DIR_NAME, COLLECTION_NAME, PARENT_STORE_DIR, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP


def get_retriever(k: int = 5, score_threshold: float = 0.5) -> ParentDocumentRetriever:
    """
    Initializes and returns the ParentDocumentRetriever.
    
    Args:
        k (int): Number of documents to retrieve.
        score_threshold (float): Minimum similarity score threshold.
        
    Returns:
        ParentDocumentRetriever: The configured retriever.
    """
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR_NAME,
    )
    
    parent_doc_store = create_kv_docstore(LocalFileStore(str(PARENT_STORE_DIR)))
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=parent_doc_store,
        child_splitter=child_splitter,
        search_type=SearchType.similarity_score_threshold,
        search_kwargs={
            "k": k,
            "score_threshold": score_threshold
        },
    )
    return retriever


def main() -> None:
    """Main execution block for the interactive retrieval test."""
    print("=== DIEM Chatbot - Document Retrieval Test ===")
    
    while True:
        try:
            query = input("\nInserisci la query di ricerca (o 'q' per uscire): ").strip()
            if query.lower() in ['q', 'quit', 'exit']:
                break
                
            if not query:
                continue
                
            threshold_input = input("Inserisci la soglia di similarità [default 0.5]: ").strip()
            score_threshold = float(threshold_input) if threshold_input else 0.5
            
            k_input = input("Inserisci il numero massimo di risultati (k) [default 5]: ").strip()
            k = int(k_input) if k_input else 5
            
            print(f"\nRicerca in corso per: '{query}' (Soglia: {score_threshold}, k: {k})...")
            
            retriever = get_retriever(k=k, score_threshold=score_threshold)
            
            # Use vectorstore similarity_search_with_score directly to get scores
            vectorstore = retriever.vectorstore
            # Embed the query to use with similarity search
            results = vectorstore.similarity_search_with_score(
                query, 
                k=k*3 # search more child chunks to find unique parents
            )

            # A simpler way to just use the retriever and show the resulting documents
            docs = retriever.invoke(query)
            
            if not docs:
                print("Nessun documento trovato che soddisfi i criteri.")
                continue
                
            print(f"\nTrovati {len(docs)} documenti.")
            print("-" * 50)
            
            for i, doc in enumerate(docs, 1):
                print(f"\n--- Documento {i} ---")
                print(f"URL: {doc.metadata.get('source', 'Sconosciuto')}")
                print(f"Lunghezza contenuto: {len(doc.page_content)} caratteri")
                print(f"Titolo/Metadati: {doc.metadata}")
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
