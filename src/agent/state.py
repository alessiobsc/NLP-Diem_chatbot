"""LangGraph state definition for the DIEM Chatbot agent."""
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class DiemState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_call_count: int      # increments each time retrieve() fires; cap = MAX_RETRIEVE_CALLS
    retrieved_context: str    # latest retrieve output, passed to generate node
    last_docs: List[Document] # latest retrieved docs, used for source URL formatting
