from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from src.utils.logger import get_logger


logger = get_logger(__name__)


class OpenRouterEmbeddings:
    """Custom Langchain Embeddings wrapper for OpenRouter API using the official OpenAI client."""
    def __init__(self, model_name: str, api_key: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Prevent API calls if the Langchain batch is empty
        if not texts:
            return []

        try:
            res = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                encoding_format="float"
            )

            # Check if res.data is present and is not None before iterating
            if hasattr(res, 'data') and res.data is not None:
                return [d.embedding if hasattr(d, 'embedding') else d['embedding'] for d in res.data]
            elif isinstance(res, dict) and res.get('data') is not None:
                return [d.get('embedding', []) for d in res['data']]

            # Log the unexpected response to help debug if OpenRouter sends a silent error
            logger.warning(f"OpenRouter API returned an unexpected response format: {res}")
            return []

        except Exception as e:
            logger.error(f"Error calling OpenRouter embeddings API: {e}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        # Handle empty query strings safely
        if not text or not text.strip():
            return []

        try:
            res = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                encoding_format="float"
            )

            # Safely check for data existence
            if hasattr(res, 'data') and res.data is not None and len(res.data) > 0:
                data = res.data[0]
                return data.embedding if hasattr(data, 'embedding') else data['embedding']
            elif isinstance(res, dict) and res.get('data') is not None and len(res['data']) > 0:
                return res['data'][0].get('embedding', [])

            logger.warning(f"OpenRouter API returned an unexpected response format for query: {res}")
            return []

        except Exception as e:
            logger.error(f"Error calling OpenRouter embedding API: {e}")
            raise e


class E5HuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """
    Custom wrapper for E5 models to automatically append
    'query: ' and 'passage: ' prefixes as required by the model.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Append "passage: " to all documents during ingestion
        formatted_texts = [f"passage: {text}" for text in texts]
        return super().embed_documents(formatted_texts)

    def embed_query(self, text: str) -> List[float]:
        # Append "query: " to the user question during retrieval
        formatted_text = f"query: {text}"
        return super().embed_query(formatted_text)