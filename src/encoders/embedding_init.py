from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_PROVIDER, OPENROUTER_EMBEDDING_MODEL, LOCAL_EMBEDDING_MODEL, OPENROUTER_API_KEY
from src.encoders.embedding_models import OpenRouterEmbeddings
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_embedding_model():
    """Builds the embedding model based on the configured provider."""
    if EMBEDDING_PROVIDER == "openrouter":
        logger.info(f"Using OpenRouter embedding model: {OPENROUTER_EMBEDDING_MODEL}")
        return OpenRouterEmbeddings(
            model_name=OPENROUTER_EMBEDDING_MODEL,
            api_key=OPENROUTER_API_KEY
        )
    elif EMBEDDING_PROVIDER == "local":
        logger.info(f"Initializing local embedding model: {LOCAL_EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}, # Or 'cuda' if available
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        raise NotImplementedError(f"EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}' is not supported. Use 'local' or 'openrouter'.")
