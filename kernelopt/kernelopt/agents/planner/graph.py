# kernelopt/kernelopt/agents/planner/graph.py
"""Planner Agent LangGraph sub-graph definition."""

from __future__ import annotations

import json

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from kernelopt.agents.schemas import OptimizationPlan
from kernelopt.agents.planner.prompts import SYSTEM_PROMPT
from kernelopt.agents.planner.tools import ALL_TOOLS, _TOOL_MAP


class PlannerState(TypedDict, total=False):
    messages: list
    plan: dict | None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def agent_node(state: PlannerState, config: RunnableConfig) -> dict:
    """Call the LLM with tools bound. Accepts a pre-built LLM via configurable for testing."""
    from kernelopt.llm import build_llm

    llm = config["configurable"].get("llm")
    if llm is None:
        llm_cfg = config["configurable"]["llm_config"]
        llm = build_llm(**llm_cfg)

    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: PlannerState, config: RunnableConfig) -> dict:
    """Execute tool calls from the last agent message.

    submit_plan is intercepted: validates the plan dict against OptimizationPlan
    and writes it to state["plan"]. All other tools are executed normally.
    """
    last = state["messages"][-1]
    tool_messages: list[ToolMessage] = []
    new_plan: dict | None = None

    for tc in last.tool_calls:
        name = tc["name"]
        args = tc["args"]
        tc_id = tc["id"]

        if name == "submit_plan":
            try:
                plan = OptimizationPlan(**args["plan"])
                new_plan = plan.model_dump()
                tool_messages.append(
                    ToolMessage(content="Plan submitted successfully.", tool_call_id=tc_id)
                )
            except Exception as exc:
                tool_messages.append(
                    ToolMessage(
                        content=f"Validation failed: {exc}. Fix the plan and resubmit.",
                        tool_call_id=tc_id,
                    )
                )

        else:
            tool_fn = _TOOL_MAP.get(name)
            if tool_fn is None:
                tool_messages.append(
                    ToolMessage(content=f"Unknown tool: {name!r}", tool_call_id=tc_id)
                )
                continue
            try:
                result = tool_fn.invoke(args, config=config)
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(result, default=str),
                        tool_call_id=tc_id,
                    )
                )
            except Exception as exc:
                tool_messages.append(
                    ToolMessage(content=f"Tool error: {exc}", tool_call_id=tc_id)
                )

    updates: dict = {"messages": tool_messages}
    if new_plan is not None:
        updates["plan"] = new_plan
    return updates


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_from_agent(state: PlannerState) -> str:
    """Route to tools if agent made tool calls, otherwise END (fallback)."""
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END
    return "tools"


def route_from_tools(state: PlannerState) -> str:
    """Route to END if plan was submitted, otherwise back to agent."""
    if state.get("plan") is not None:
        return END
    return "agent"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def compile_planner_graph():
    """Return a compiled LangGraph sub-graph for one planner instance."""
    graph = StateGraph(PlannerState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_from_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_from_tools, {"agent": "agent", END: END})

    return graph.compile()


planner_subgraph = compile_planner_graph()
