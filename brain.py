"""
Core AI Brain module for the DIEM Chatbot.
Handles the initialization of the Language Model, Document Retriever, and RAG Pipeline.
"""

from operator import itemgetter
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_vector import SearchType

from sentence_transformers import CrossEncoder

from config import (
    OLLAMA_CHAT_MODEL,
    LLM_TEMPERATURE,
    PARENT_STORE_DIR,
    EMBEDDING_MODEL_NAME,
    CROSS_ENCODER_MODEL_NAME,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    BI_ENCODER_K,
    CROSS_ENCODER_K,
    RETRIEVER_SCORE_THRESHOLD,
    DEFAULT_SESSION_ID
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared embedding model (used by both ingestion and app)
# ─────────────────────────────────────────────────────────────────────────────
embedding_model = HuggingFaceEmbeddings( # type: ignore
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},
)

# ─────────────────────────────────────────────────────────────────────────────
# Cross-Encoder Reranker model
# ─────────────────────────────────────────────────────────────────────────────
reranker = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

def rerank(query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
    """
    Reranks documents using a Cross-Encoder model to find the most relevant ones.
    
    Args:
        query (str): The search query.
        documents (List[Document]): The initial list of retrieved documents.
        top_n (int): The number of top documents to return.

    Returns:
        List[Document]: Reranked and filtered documents.
    """
    if not documents:
        return []
        
    pairs = [[query, d.page_content] for d in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    out = []
    for d, s in ranked[:top_n]:
        d.metadata["relevance_score"] = float(s)
        out.append(d)
    return out

class DiemBrain:
    """
    Encapsulates the conversational Retrieval-Augmented Generation (RAG) system for the DIEM department.

    This class is responsible for:
    1. Setting up the LLM and prompt templates.
    2. Configuring the document retriever.
    3. Building the execution chain with query rewriting and history management.
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant for the DIEM department "
        "(Department of Information and Electrical Engineering and Applied Mathematics) "
        "at the University of Salerno, Italy. "
        "Answer questions using ONLY the provided context. Do not use prior knowledge. "
        "If the answer is not in the context, say: "
        "'I don't have that information in my knowledge base.' "
        "If the question is unrelated to DIEM or the University of Salerno, say: "
        "'This question is outside my scope. I can only answer questions about DIEM.'"
    )

    REWRITE_PROMPT = (
        "Rewrite the user's question into a self-contained search query using the chat history. "
        "Resolve all pronouns and references (e.g. 'it', 'they', 'that professor'). "
        "Return ONLY the rewritten query."
    )

    def __init__(self, vectorstore: Chroma) -> None:
        """
        Initializes the DiemBrain with the specified vector store.

        Args:
            vectorstore (Chroma): The Chroma vector store used for document retrieval.
        """
        self._chat_model = ChatOllama(
            model=OLLAMA_CHAT_MODEL,
            temperature=LLM_TEMPERATURE,
        )
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

    def _build_retriever(self, vectorstore: Chroma) -> ParentDocumentRetriever:
        """
        Constructs the Parent Document Retriever.

        Args:
            vectorstore (Chroma): The underlying vector store.

        Returns:
            ParentDocumentRetriever: Configured retriever instance.
        """
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
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("placeholder", "{history}"),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"),
        ])

        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", self.REWRITE_PROMPT),
            ("placeholder", "{history}"),
            ("human", "{question}"),
        ])

        # Chain to rewrite the query using context from history
        rewrite_chain = (
            rewrite_prompt
            | self._chat_model
            | RunnableLambda(lambda m: m.content.strip())
        )

        # Skip rewrite if history is empty
        conditional_rewrite_chain = RunnableBranch(
            (
                lambda x: not x.get("history"),
                itemgetter("question")
            ),
            rewrite_chain
        )

        self._rag_prompt = rag_prompt
        self._rewrite_chain = conditional_rewrite_chain

        # Core RAG logic: Retrieve -> Rerank -> Format -> Generate
        rag_chain = (
            {
                "docs": itemgetter("question") | retriever,
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
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
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
            self._store[session_id] = InMemoryChatMessageHistory()
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
            return ""

        seen_urls = set()
        unique_urls = []
        
        for doc in sources:
            url = doc.metadata.get("source", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_urls.append(url)
                
        if unique_urls:
            return "\n\n**Sources:**\n" + "\n".join(f"- {url}" for url in unique_urls)
            
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
        try:
            result = self.conversational_rag.invoke(
                {"question": message},
                config={"configurable": {"session_id": session_id}},
            )
            
            answer: str = result["answer"]
            sources: List[Document] = result.get("sources", [])
            
            formatted_sources = self._format_sources(sources)
            return answer + formatted_sources
            
        except Exception as e:
            print(f"Error during chat processing: {e}")
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

        context = "\n\n---\n\n".join(d.page_content for d in reranked)
        prompt_value = self._rag_prompt.invoke({
            "context": context,
            "question": rewritten,
            "history": history_messages,
        })

        answer = ""
        try:
            for chunk in self._chat_model.stream(prompt_value):
                answer += chunk.content
                yield answer
        except Exception as e:
            print(f"Error during streaming: {e}")
            yield "Mi dispiace, si è verificato un errore durante la generazione della risposta."
            return

        history.add_user_message(message)
        history.add_ai_message(answer)

        sources_md = self._format_sources(reranked)
        if sources_md:
            yield answer + sources_md