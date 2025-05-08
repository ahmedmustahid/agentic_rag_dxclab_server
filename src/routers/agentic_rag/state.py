from typing import Annotated, Any, Dict

from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


class State(MessagesState):
    messages: Annotated[list, add_messages]
    turn: int
    request: str  # User question or request
    rev_request: str  # Revised user question or request
    plan: Dict[str, Any]
    plan_status: list
    plan_over: bool
    plan_exec: Dict[str, Any]  # Plan to execute
