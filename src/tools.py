"""
RAG tools for the agentic DIEM Chatbot.

Provides four composable tools:
- retrieve: Search the DIEM knowledge base
- summarize: Summarize long text
- calculate: Apply academic calculations using retrieved formulas
- answer: Generate final answer using retrieved context
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.tools import tool

from config import CROSS_ENCODER_K
from src.logger import get_logger

logger = get_logger(__name__)


def build_tools(retriever, generation_model, brain_ref, rag_prompt) -> list:
    """
    Build the 4 RAG tools. brain_ref._last_docs is updated by retrieve()
    so DiemBrain.chat() can access retrieved documents after agent completes.

    Args:
        retriever: Document retriever (LangChain Retriever)
        generation_model: Language model for generation
        brain_ref: Reference to DiemBrain instance (stores _last_docs)
        rag_prompt: RAG prompt template (currently unused but kept for API consistency)

    Returns:
        List[tool]: List of 4 tool functions
    """

    @tool
    def retrieve(query: str) -> str:
        """Search the DIEM knowledge base for documents relevant to the query.
        Always call this before answer() or calculate()."""
        from src.brain import rerank, _format_context

        docs = retriever.invoke(query)
        reranked = rerank(query, docs, top_n=CROSS_ENCODER_K) if docs else []
        brain_ref._last_docs = reranked
        logger.info(f"retrieve: {len(reranked)} docs after rerank")
        return _format_context({"docs": reranked, "question": query, "history": []})["context"]

    @tool
    def summarize(text: str) -> str:
        """Summarize long text into concise key points. Use when retrieved context is very long."""
        return generation_model.invoke(f"Summarize concisely in Italian:\n{text}").content

    @tool
    def calculate(context: str, operation: str, values: dict) -> str:
        """Apply an academic calculation using a formula already retrieved from the knowledge base.
        Call retrieve() first to fetch the official DIEM formula, then pass its output as context.
        Use for: graduation grade from average, weighted average, TOLC score thresholds."""
        prompt = (
            f"Using only the official formula found in the following context, "
            f"compute the result for: operation='{operation}', values={values}.\n\n"
            f"Context:\n{context}\n\n"
            f"Show the calculation steps and the final result. "
            f"If the formula is not found in the context, say so explicitly."
        )
        return generation_model.invoke(prompt).content

    @tool
    def answer(context: str, question: str) -> str:
        """Generate the final answer to the user question using retrieved context.
        ALWAYS call retrieve() first, then pass its output as context here.
        Never call this without context from retrieve()."""
        from src.prompts import SYSTEM_PROMPT
        from langchain_core.prompts import ChatPromptTemplate

        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "<context>\n{context}\n</context>\n\n<instruction>\n{question}\n</instruction>"),
        ])
        return generation_model.invoke(
            prompt_tmpl.invoke({"context": context, "question": question})
        ).content

    return [retrieve, summarize, calculate, answer]
