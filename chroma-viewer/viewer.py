import chromadb
from chromadb.config import Settings
import pandas as pd
import streamlit as st
import os

pd.set_option('display.max_columns', 4)

def view_collections(dir):
    """
    Connects to a Chroma DB persistent client and displays its collections and documents.

    Args:
        dir (str): The path to the Chroma DB directory.
    """
    if not os.path.isdir(dir):
        st.error(f"Error: Directory not found at '{dir}'")
        return

    st.markdown(f"### DB Path: {dir}")

    try:
        client = chromadb.PersistentClient(path=dir)
    except Exception as e:
        st.error(f"Failed to connect to Chroma DB: {e}")
        return

    st.header("Collections")

    collections = client.list_collections()
    if not collections:
        st.warning("No collections found in this database.")
        return

    for collection in collections:
        try:
            # Count total documents to show the user
            total_docs = collection.count()
            st.markdown(f"### Collection: **{collection.name}** (Total Documents: {total_docs})")

            if total_docs == 0:
                st.info("This collection is empty.")
                continue

            # The get() method retrieves all data if no limit is specified.
            # We exclude 'embeddings' to avoid Streamlit MessageSizeError (data exceeding 200MB limit).
            # Embeddings are large vectors and usually not useful to view as raw text.
            data = collection.get(
                include=["metadatas", "documents"]
            )

            # Estrai i dati, gestendo il caso in cui ChromaDB restituisca None
            documents = data.get("documents")
            documents = documents if documents is not None else [None] * len(data.get("ids", []))

            metadatas = data.get("metadatas")
            metadatas = metadatas if metadatas is not None else [None] * len(data.get("ids", []))
            
            # Extract the 'source' from each metadata dictionary.
            # If a metadata object is None or 'source' key doesn't exist, it defaults to 'N/A'.
            sources = [
                meta.get("source", "N/A") if meta else "N/A" 
                for meta in metadatas
            ]

            # Create a DataFrame from the retrieved data
            df = pd.DataFrame({
                "Source": sources,
                "Document": documents,
                "Metadata": metadatas
            })

            # Streamlit displays dataframes efficiently, but sending >200MB at once will crash it.
            # Without embeddings, the dataframe size is drastically reduced.
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to retrieve data for collection '{collection.name}': {e}")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Chroma DB Viewer")

    # Get database path from user input
    db_path = st.text_input("Enter the path to your Chroma DB directory:", "")

    if db_path:
        view_collections(db_path)
