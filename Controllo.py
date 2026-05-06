from langchain_chroma import Chroma
from brain import embedding_model # Assicurati che il nome sia corretto

vectorstore = Chroma(
    collection_name="diem_knowledge",
    persist_directory="chroma_diem",
    embedding_function=embedding_model
)


results = vectorstore.get(limit=10, include=['documents', 'metadatas'])

for i in range(len(results['documents'])):
    print(f"--- Documento {i+1} ---")
    print(f"SOURCE: {results['metadatas'][i].get('source')}")
    print(f"CONTENT: {results['documents'][i][:200]}...\n")