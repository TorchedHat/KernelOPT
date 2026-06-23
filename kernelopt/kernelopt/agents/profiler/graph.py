# kernelopt/kernelopt/agents/profiler/graph.py
"""Profiler Agent LangGraph sub-graph definition."""

from __future__ import annotations

import json
from typing import Annotated, Optional

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from kernelopt.agents.profiler.prompts import SYSTEM_PROMPT
from kernelopt.agents.profiler.tools import ALL_TOOLS, _TOOL_MAP


class ProfilerState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    ncu_config: Optional[dict]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def agent_node(state: ProfilerState, config: RunnableConfig) -> dict:
    """Call the LLM with tools bound."""
    from kernelopt.llm import build_llm

    llm = config["configurable"].get("llm")
    if llm is None:
        llm_cfg = config["configurable"]["llm_config"]
        llm = build_llm(**llm_cfg)

    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: ProfilerState, config: RunnableConfig) -> dict:
    """Execute tool calls from the last agent message.

    submit_ncu_config is intercepted: validates the config dict and writes it
    to state["ncu_config"]. read_kernel_source is executed via _TOOL_MAP.
    """
    last = state["messages"][-1]
    tool_messages: list[ToolMessage] = []
    new_ncu_config: Optional[dict] = None

    for tc in last.tool_calls:
        name = tc["name"]
        args = tc["args"]
        tc_id = tc["id"]

        if name == "submit_ncu_config":
            kernel_name = args.get("kernel_name", "")
            launch_skip = args.get("launch_skip", 0)
            launch_count = args.get("launch_count", 1)
            kernel_args = args.get("kernel_args", [])
            replay_mode = args.get("replay_mode", "kernel")

            errors = []
            if not isinstance(kernel_name, str) or not kernel_name.strip():
                errors.append("kernel_name must be a non-empty string")
            if not isinstance(launch_skip, int) or launch_skip < 0:
                errors.append("launch_skip must be int >= 0")
            if launch_count is not None and (not isinstance(launch_count, int) or launch_count < 1):
                errors.append("launch_count must be int >= 1 or null")
            if replay_mode not in ("kernel", "application"):
                errors.append("replay_mode must be 'kernel' or 'application'")

            if errors:
                tool_messages.append(
                    ToolMessage(
                        content=f"Validation failed: {'; '.join(errors)}. Fix and resubmit.",
                        tool_call_id=tc_id,
                    )
                )
            else:
                new_ncu_config = {
                    "kernel_name": kernel_name.strip(),
                    "launch_skip": launch_skip,
                    "launch_count": launch_count,
                    "kernel_args": list(kernel_args),
                    "replay_mode": replay_mode,
                }
                tool_messages.append(
                    ToolMessage(
                        content="NCU config submitted successfully.",
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
                        content=json.dumps(result, default=str)
                        if not isinstance(result, str)
                        else result,
                        tool_call_id=tc_id,
                    )
                )
            except Exception as exc:
                tool_messages.append(
                    ToolMessage(content=f"Tool error: {exc}", tool_call_id=tc_id)
                )

    updates: dict = {"messages": tool_messages}
    if new_ncu_config is not None:
        updates["ncu_config"] = new_ncu_config
    return updates


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_from_agent(state: ProfilerState) -> str:
    """Route to tools if agent made tool calls, otherwise END (fallback)."""
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END
    return "tools"


def route_from_tools(state: ProfilerState) -> str:
    """Route to END if ncu_config was submitted, otherwise back to agent."""
    if state.get("ncu_config") is not None:
        return END
    return "agent"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def compile_profiler_graph():
    """Return a compiled LangGraph sub-graph for the profiler config-decider."""
    graph = StateGraph(ProfilerState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_from_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_from_tools, {"agent": "agent", END: END})

    return graph.compile()


profiler_subgraph = compile_profiler_graph()
