from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state passed between every node in the graph.
    LangGraph merges each node's return dict into this state automatically,
    using the reducer specified per field (or overwrite, if none specified).
    """
    messages: Annotated[list, add_messages]
    topic: str
    research_plan: str
    collected_data: list[str]
    sources: list[str]
    iteration_count: int
    next_step: str