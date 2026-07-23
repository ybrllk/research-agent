from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from nodes import planner_node, researcher_node, responder_node


def route_after_planner(state: AgentState) -> str:
    """
    Conditional edge function. Reads the 'next_step' field the Planner
    set and routes to the matching node name.
    """
    return "researcher" if state.get("next_step") == "research" else "responder"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("responder", responder_node)

    graph.set_entry_point("planner")

    # Conditional edge: Planner's output decides the next node.
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"researcher": "researcher", "responder": "responder"},
    )

    # Researcher always loops back to Planner, which re-checks the
    # iteration count guard on every pass.
    graph.add_edge("researcher", "planner")

    graph.add_edge("responder", END)

    # MemorySaver = in-memory checkpointer. State persists across turns
    # within the same process, keyed by thread_id.
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)