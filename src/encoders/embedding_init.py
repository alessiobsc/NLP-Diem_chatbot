from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_PROVIDER, OPENROUTER_EMBEDDING_MODEL, LOCAL_EMBEDDING_MODEL, OPENROUTER_API_KEY
from src.encoders.embedding_models import OpenRouterEmbeddings, E5HuggingFaceEmbeddings
from src.utils.logger import get_logger


logger = get_logger(__name__)



def build_embedding_model():
    """Builds the embedding model based on the configured provider."""
    if EMBEDDING_PROVIDER == "openrouter":
        logger.info(f"Using OpenRouter embedding model: {OPENROUTER_EMBEDDING_MODEL}")
        query_instruction = (
            "Retrieve relevant passages that answer the user's query "
            if "qwen3" in OPENROUTER_EMBEDDING_MODEL.lower() else ""
        )
        return OpenRouterEmbeddings(
            model_name=OPENROUTER_EMBEDDING_MODEL,
            api_key=OPENROUTER_API_KEY,
            query_instruction=query_instruction,
        )
    elif EMBEDDING_PROVIDER == "local":
        # Use local model directly (with prefix logic for e5 models)
        logger.info(f"Initializing local embedding model: {LOCAL_EMBEDDING_MODEL}")

        if "e5" in LOCAL_EMBEDDING_MODEL.lower():
            return E5HuggingFaceEmbeddings(
                model_name=LOCAL_EMBEDDING_MODEL,
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            return HuggingFaceEmbeddings(
                model_name=LOCAL_EMBEDDING_MODEL,
                encode_kwargs={"normalize_embeddings": True},
            )
    else:
        raise NotImplementedError(f"EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}' is not supported. Use 'local' or 'openrouter'.")