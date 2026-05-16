"""
Core AI Brain module for the DIEM Chatbot.
Handles the initialization of the Language Model, Document Retriever, and RAG Pipeline.
"""

from operator import itemgetter
from typing import Any, Dict, List
import requests
import json
from openai import OpenAI

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_vector import SearchType

from sentence_transformers import CrossEncoder

from config import (
    LLM_PROVIDER,
    EMBEDDING_PROVIDER,
    OLLAMA_CHAT_MODEL,
    LLM_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    PARENT_STORE_DIR,
    OPENROUTER_EMBEDDING_MODEL,
    OPENROUTER_RERANKER_MODEL,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_RERANKER_MODEL,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    BI_ENCODER_K,
    CROSS_ENCODER_K,
    RETRIEVER_SCORE_THRESHOLD,
    DEFAULT_SESSION_ID
)

from src.prompts import SYSTEM_PROMPT, REWRITE_PROMPT, REJECTION_TAGS
from src.logger import get_logger

logger = get_logger(__name__)


def _build_chat_model():
    if LLM_PROVIDER == "openrouter":
        logger.info(f"Using OpenRouter LLM: {OPENROUTER_MODEL} (temp={LLM_TEMPERATURE})")
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            model=OPENROUTER_MODEL,
            temperature=LLM_TEMPERATURE,
        )
    elif LLM_PROVIDER == "local":
        logger.info(f"Using Ollama LLM: {OLLAMA_CHAT_MODEL} (temp={LLM_TEMPERATURE})")
        return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=LLM_TEMPERATURE)
    else:
        raise NotImplementedError(f"LLM_PROVIDER '{LLM_PROVIDER}' is not supported. Use 'local' or 'openrouter'.")


# ─────────────────────────────────────────────────────────────────────────────
# Shared embedding model (used by both ingestion and app)
# ─────────────────────────────────────────────────────────────────────────────

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


def _build_embedding_model():
    """Builds the embedding model based on the configured provider."""
    if EMBEDDING_PROVIDER == "openrouter":
        logger.info(f"Using OpenRouter embedding model: {OPENROUTER_EMBEDDING_MODEL}")
        return OpenRouterEmbeddings(
            model_name=OPENROUTER_EMBEDDING_MODEL, 
            api_key=OPENROUTER_API_KEY
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

embedding_model = _build_embedding_model()

# ─────────────────────────────────────────────────────────────────────────────
# Reranker model
# ─────────────────────────────────────────────────────────────────────────────

def _rerank_with_openrouter(query: str, documents: List[Document], top_n: int) -> List[Document]:
    """Reranks documents using the official OpenRouter rerank endpoint."""
    if not documents:
        return []
        
    logger.debug(f"Reranking {len(documents)} documents for query: '{query}' with OpenRouter: {OPENROUTER_RERANKER_MODEL}")

    docs_content = [d.page_content for d in documents]

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/rerank",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": OPENROUTER_RERANKER_MODEL,
                "query": query,
                "documents": docs_content,
                "top_n": top_n
            }),
            timeout=15
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        
        if not results:
             raise ValueError("No results returned from OpenRouter rerank API.")

        reranked_docs = []
        for result in results:
            doc = documents[result["index"]]
            score = result["relevance_score"]
            doc.metadata["relevance_score"] = score
            logger.debug(f"Reranked doc (score={score:.4f}): {doc.metadata.get('source', 'Unknown')}")
            reranked_docs.append(doc)
        
        logger.info(f"Selected top {len(reranked_docs)} documents after OpenRouter reranking")
        return reranked_docs

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling OpenRouter rerank API: {e}")
        raise e


def _rerank_local(query: str, documents: List[Document], top_n: int) -> List[Document]:
    """Reranks documents using a local Cross-Encoder model."""
    if not documents:
        return []
    
    logger.debug(f"Reranking {len(documents)} documents for query: '{query}' with local model: {LOCAL_RERANKER_MODEL}")
    
    try:
        reranker = CrossEncoder(LOCAL_RERANKER_MODEL)
    except Exception as e:
        logger.error(f"Failed to load local reranker model '{LOCAL_RERANKER_MODEL}': {e}")
        raise e

    pairs = [[query, d.page_content] for d in documents]
    scores = reranker.predict(pairs, show_progress_bar=False)
    
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    out = []
    for i, (d, s) in enumerate(ranked[:top_n]):
        d.metadata["relevance_score"] = float(s)
        logger.debug(f"Reranked rank {i+1}: score={s:.4f}, source={d.metadata.get('source', 'Unknown')}")
        out.append(d)
    
    logger.info(f"Selected top {len(out)} documents after local reranking")
    return out


def rerank(query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
    """Dispatches to the appropriate reranking function based on the provider."""
    if EMBEDDING_PROVIDER == "openrouter":
        return _rerank_with_openrouter(query, documents, top_n)
    elif EMBEDDING_PROVIDER == "local":
        return _rerank_local(query, documents, top_n)
    else:
        raise NotImplementedError(f"EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}' is not supported. Use 'local' or 'openrouter'.")


class DiemBrain:
    """
    Encapsulates the conversational Retrieval-Augmented Generation (RAG) system for the DIEM department.

    This class is responsible for:
    1. Setting up the LLM and prompt templates.
    2. Configuring the document retriever.
    3. Building the execution chain with query rewriting and history management.
    """

    def __init__(self, vectorstore: Chroma) -> None:
        """
        Initializes the DiemBrain with the specified vector store.

        Args:
            vectorstore (Chroma): The Chroma vector store used for document retrieval.
        """
        self._chat_model = _build_chat_model()
        self._store: Dict[str, InMemoryChatMessageHistory] = {}
        
        self._retriever = self._build_retriever(vectorstore)
        rag_chain = self._build_rag_chain(self._retriever)
        
        self.conversational_rag = RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="question",
            output_messages_key="answer",
            history_messages_key="history",
        )
        logger.info("DiemBrain initialization complete")

    def _build_retriever(self, vectorstore: Chroma) -> ParentDocumentRetriever:
        """
        Constructs the Parent Document Retriever.

        Args:
            vectorstore (Chroma): The underlying vector store.

        Returns:
            ParentDocumentRetriever: Configured retriever instance.
        """
        logger.debug(f"Building ParentDocumentRetriever (chunk_size={CHILD_CHUNK_SIZE}, overlap={CHILD_CHUNK_OVERLAP})")
        parent_doc_store = create_kv_docstore(LocalFileStore(str(PARENT_STORE_DIR)))
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE, 
            chunk_overlap=CHILD_CHUNK_OVERLAP
        )
        
        return ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=parent_doc_store,
            child_splitter=child_splitter,
            search_type=SearchType.similarity_score_threshold,
            search_kwargs={
                "k": BI_ENCODER_K,
                "score_threshold": RETRIEVER_SCORE_THRESHOLD
            },
        )

    def _build_rag_chain(self, retriever: ParentDocumentRetriever) -> Runnable:
        """
        Builds the complete RAG chain including query rewriting, retrieval, cross-encoder reranking, and response generation.

        Args:
            retriever (ParentDocumentRetriever): The configured document retriever.

        Returns:
            Runnable: The executable LangChain runnable sequence.
        """
        logger.debug("Building RAG chain")
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{history}"),
            ("human", "<context>\n{context}\n</context>\n\n<instruction>\n{question}\n</instruction>"),
        ])

        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITE_PROMPT),
            ("placeholder", "{history}"),
            ("human", "{question}"),
        ])

        def _log_rewrite(q: str) -> str:
            logger.info(f"Rewritten query using history: '{q}'")
            return q
            
        def _log_skip_rewrite(x: Dict[str, Any]) -> str:
            logger.debug(f"Skipping query rewrite (empty history). Original query: '{x['question']}'")
            return x["question"]

        def _log_retrieved_docs(docs: List[Document]) -> List[Document]:
            logger.info(f"Retriever fetched {len(docs)} documents")
            return docs

        # Chain to rewrite the query using context from history
        rewrite_chain = (
            rewrite_prompt
            | self._chat_model
            | RunnableLambda(lambda m: m.content.strip())
            | RunnableLambda(_log_rewrite)
        )

        # Skip rewrite if history is empty
        conditional_rewrite_chain = RunnableBranch(
            (
                lambda x: not x.get("history"),
                RunnableLambda(_log_skip_rewrite)
            ),
            rewrite_chain
        )

        self._rag_prompt = rag_prompt
        self._rewrite_chain = conditional_rewrite_chain

        # Core RAG logic: Retrieve -> Rerank -> Format -> Generate
        rag_chain = (
            {
                "docs": itemgetter("question") | retriever | RunnableLambda(_log_retrieved_docs),
                "question": itemgetter("question"),
                "history": itemgetter("history"),
            }
            # Add the reranking step: passing the retrieved docs and query to the cross-encoder
            | RunnablePassthrough.assign(
                docs=lambda x: rerank(x["question"], x["docs"], top_n=CROSS_ENCODER_K) if x["docs"] else []
            )
            | RunnableLambda(self._format_context)
            | RunnableLambda(lambda x: {
                "answer": self._chat_model.invoke(
                    rag_prompt.invoke({
                        "context": x["context"],
                        "question": x["question"],
                        "history": x["history"],
                    })
                ).content,
                "sources": x["docs"],
            })
        )

        # Combine rewrite and core RAG pipeline
        return (
            RunnablePassthrough()
            | RunnablePassthrough.assign(question=conditional_rewrite_chain)
            | rag_chain
        )


    @staticmethod
    def _format_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats retrieved documents into a single context string.

        Args:
            inputs (Dict[str, Any]): Dictionary containing 'docs'.

        Returns:
            Dict[str, Any]: Inputs augmented with the 'context' string.
        """
        docs: List[Document] = inputs.get("docs", [])
        logger.debug(f"Formatting context from {len(docs)} reranked documents")

        formatted_docs = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown Source")
            block = (
                "<document>\n"
                f"<source>{source}</source>\n"
                f"<content>\n{doc.page_content}\n</content>\n"
                "</document>"
            )
            formatted_docs.append(block)

        context = "\n\n".join(formatted_docs)
        if docs:
            logger.debug(f"Total formatted context length: {len(context)} characters")
        return {**inputs, "context": context}


    def _get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """
        Retrieves or creates the chat history for a given session ID.

        Args:
            session_id (str): The unique identifier for the chat session.

        Returns:
            InMemoryChatMessageHistory: The chat history object.
        """
        if session_id not in self._store:
            logger.info(f"Creating new chat history for session ID: {session_id}")
            self._store[session_id] = InMemoryChatMessageHistory()
        else:
            logger.debug(f"Retrieved existing chat history for session ID: {session_id}")
            logger.debug(f"Current history length: {len(self._store[session_id].messages)} messages")
        return self._store[session_id]


    def _format_sources(self, sources: List[Document]) -> str:
        """
        Extracts and formats unique source URLs from retrieved documents.

        Args:
            sources (List[Document]): The list of retrieved documents.

        Returns:
            str: Formatted markdown string of sources, or empty string if none.
        """
        if not sources:
            logger.debug("No sources found to format")
            return ""

        seen_urls = set()
        unique_urls = []
        
        for doc in sources:
            url = doc.metadata.get("source", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_urls.append(url)
                
        if unique_urls:
            logger.debug(f"Extracted {len(unique_urls)} unique source URLs out of {len(sources)} documents")
            return "\n\n**Sources:**\n" + "\n".join(f"- {url}" for url in unique_urls)
            
        logger.debug("Documents had no valid source metadata")
        return ""


    def chat(self, message: str, session_id: str = DEFAULT_SESSION_ID) -> str:
        """
        Processes a user message and returns the chatbot's response.

        Args:
            message (str): The user's input message.
            session_id (str, optional): The session identifier. Defaults to DEFAULT_SESSION_ID.

        Returns:
            str: The generated response, potentially including sources.
        """
        logger.info(f"--- New chat request | Session: {session_id} ---")
        logger.debug(f"User message: '{message}'")
        try:
            result = self.conversational_rag.invoke(
                {"question": message},
                config={"configurable": {"session_id": session_id}},
            )
            
            answer: str = result["answer"]
            sources: List[Document] = result.get("sources", [])

            logger.info("Chat message processed successfully")
            logger.debug(f"Generated answer (length={len(answer)}): {answer[:100]}...")

            is_rejection = any(answer.startswith(t) for t in REJECTION_TAGS)
            if is_rejection:
                for tag in REJECTION_TAGS:
                    if answer.startswith(tag):
                        answer = answer[len(tag):].lstrip()
                        break
            else:
                answer += self._format_sources(sources)
            return answer
            
        except Exception as e:
            logger.exception(f"Error during chat processing: {e}")
            return "Mi dispiace, si è verificato un errore durante l'elaborazione della tua richiesta."

    def chat_stream(self, message: str, session_id: str = DEFAULT_SESSION_ID):
        """
        Streaming generator: runs full RAG pipeline silently, then yields LLM tokens incrementally.
        Bypasses RunnableWithMessageHistory and manages history manually.
        """
        history = self._get_history(session_id)
        history_messages = history.messages

        rewritten = self._rewrite_chain.invoke({
            "question": message,
            "history": history_messages,
        }) if history_messages else message

        docs = self._retriever.invoke(rewritten)
        reranked = rerank(rewritten, docs, top_n=CROSS_ENCODER_K) if docs else []

        formatted = self._format_context({"docs": reranked, "question": rewritten, "history": history_messages})
        context = formatted["context"]
        prompt_value = self._rag_prompt.invoke({
            "context": context,
            "question": rewritten,
            "history": history_messages,
        })

        _max_tag_len = max(len(t) for t in REJECTION_TAGS)
        answer = ""
        tag_checked = False
        is_rejection = False

        try:
            for chunk in self._chat_model.stream(prompt_value):
                answer += chunk.content
                if not tag_checked and len(answer) >= _max_tag_len + 2:
                    tag_checked = True
                    for tag in REJECTION_TAGS:
                        if answer.startswith(tag):
                            is_rejection = True
                            answer = answer[len(tag):].lstrip()
                            break
                if tag_checked:
                    yield answer
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield "Mi dispiace, si è verificato un errore durante la generazione della risposta."
            return

        if not tag_checked:
            for tag in REJECTION_TAGS:
                if answer.startswith(tag):
                    is_rejection = True
                    answer = answer[len(tag):].lstrip()
                    break
            if answer:
                yield answer

        history.add_user_message(message)
        history.add_ai_message(answer)

        if not is_rejection:
            sources_md = self._format_sources(reranked)
            if sources_md:
                yield answer + sources_md