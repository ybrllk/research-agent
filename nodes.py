import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from state import AgentState
from tools import web_search, document_retrieval

load_dotenv()

MAX_ITERATIONS = 3

# Plain model for reasoning steps (Planner, Responder) — no tools bound.
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Separate instance with tools bound — only the Researcher node uses this.
llm_with_tools = llm.bind_tools([web_search, document_retrieval])

TOOLS_BY_NAME = {"web_search": web_search, "document_retrieval": document_retrieval}


def planner_node(state: AgentState) -> dict:
    """
    Decides whether the agent has enough information to answer, or needs
    another research pass. Enforces the max-iteration guard so Planner and
    Researcher can never loop forever.
    """
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return {
            "next_step": "respond",
            "research_plan": "Max research iterations reached — answering with what's available.",
        }

    topic = state["topic"]
    collected = "\n".join(state.get("collected_data", [])) or "(nothing yet)"

    prompt = (
        f"You are a research planner. The user's topic is: '{topic}'.\n\n"
        f"Information collected so far:\n{collected}\n\n"
        "Decide if this is enough to write a well-supported answer. "
        "Reply in EXACTLY this format, nothing else:\n"
        "DECISION: RESEARCH\nPLAN: <one sentence on what's still missing>\n"
        "OR\n"
        "DECISION: RESPOND"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    # Explicit parsing, falls back to RESPOND on anything unexpected —
    # a malformed LLM reply can never cause an infinite loop.
    if text.startswith("DECISION: RESEARCH"):
        plan_line = next((l for l in text.splitlines() if l.startswith("PLAN:")), "PLAN: gather more sources")
        return {"next_step": "research", "research_plan": plan_line.replace("PLAN:", "").strip()}

    return {"next_step": "respond", "research_plan": "Sufficient information collected."}


def researcher_node(state: AgentState) -> dict:
    """
    Calls tools to gather information for the current research plan.
    The LLM decides which tool(s) to call via bind_tools; this code
    executes those calls and merges results into state.
    """
    topic = state["topic"]
    plan = state.get("research_plan", "")

    prompt = (
        f"Research topic: {topic}\n"
        f"What's still needed: {plan}\n\n"
        "Use the available tools to find relevant information."
    )

    try:
        response = llm_with_tools.invoke([HumanMessage(content=prompt)])
        tool_calls = getattr(response, "tool_calls", None) or []
    except Exception as e:
        # Error case: the model/provider failed to produce a valid tool
        # call (a known quirk with Groq + Llama's function-calling format).
        # Don't crash the whole run — fall back to a direct search instead.
        print(f"  [warning] tool-calling request failed ({e}); falling back to web_search")
        tool_calls = []

    new_data = list(state.get("collected_data", []))
    new_sources = list(state.get("sources", []))

    if not tool_calls:
        # Either the model skipped calling a tool, or the request above
        # failed — force progress by calling web_search directly.
        tool_calls = [{"name": "web_search", "args": {"query": topic}, "id": "fallback"}]

    for call in tool_calls:
        tool_fn = TOOLS_BY_NAME.get(call["name"])
        try:
            result = tool_fn.invoke(call["args"]) if tool_fn else "Error: unknown tool requested."
        except Exception as e:
            result = f"Error: tool '{call['name']}' failed ({e})."
        print(f"  [tool call] {call['name']}({call['args']}) -> {result[:80]}...")
        new_data.append(result)
        new_sources.append(call["name"])

    return {
        "collected_data": new_data,
        "sources": new_sources,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def responder_node(state: AgentState) -> dict:
    """
    Synthesizes a final answer from everything gathered, keeping source
    attribution intact, and appends it to the conversation.
    """
    topic = state["topic"]
    collected = "\n".join(state.get("collected_data", [])) or "No information was found."

    prompt = (
        f"Research topic: {topic}\n\n"
        f"Gathered findings (each already includes its source tag):\n{collected}\n\n"
        "Write a clear, well-organized answer to the topic using this information. "
        "Keep each source's [Source: ...] tag attached to the claim it supports."
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [AIMessage(content=response.content)]}