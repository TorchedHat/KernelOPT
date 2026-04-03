# kernelopt/kernelopt/agents/executor/graph.py
"""Executor Agent LangGraph sub-graph definition."""

from __future__ import annotations

import difflib
import json
from typing import Annotated

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from kernelopt.agents.executor.prompts import SYSTEM_PROMPT
from kernelopt.agents.executor.tools import ALL_TOOLS, _TOOL_MAP


class ExecutorState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    kernel_result: dict | None   # set on success or K exhaustion; None until then
    attempt: int                 # number of submit_kernel calls made so far


_KERNEL_MARKERS = ["@triton.jit", "__global__", "@helion.kernel", "@hl.kernel"]


def _infer_framework(kernel_source: str) -> str:
    if "@triton.jit" in kernel_source:
        return "triton"
    if "__global__" in kernel_source:
        return "cuda"
    if "@helion.kernel" in kernel_source or "@hl.kernel" in kernel_source:
        return "helion"
    return "unknown"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def agent_node(state: ExecutorState, config: RunnableConfig) -> dict:
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


def tool_node(state: ExecutorState, config: RunnableConfig) -> dict:
    """Execute tool calls from the last agent message.

    submit_kernel is intercepted: validates the kernel source, increments the attempt
    counter, and writes kernel_result to state on success or K-exhaustion.
    All other tools are executed normally via _TOOL_MAP.
    """
    last = state["messages"][-1]
    K: int = config["configurable"]["K"]
    plan_id: str = config["configurable"]["plan_id"]
    original_kernel: str = config["configurable"]["original_kernel"]

    tool_messages: list[ToolMessage] = []
    new_kernel_result: dict | None = None
    new_attempt: int = state.get("attempt", 0)

    for tc in last.tool_calls:
        name = tc["name"]
        args = tc["args"]
        tc_id = tc["id"]

        if name == "submit_kernel":
            new_attempt = state.get("attempt", 0) + 1
            kernel_source: str = args.get("kernel_source", "")
            change_summary: str = args.get("change_summary", "")

            has_marker = any(m in kernel_source for m in _KERNEL_MARKERS)
            is_valid = bool(kernel_source.strip()) and has_marker

            if is_valid:
                framework = _infer_framework(kernel_source)
                lines_changed = sum(
                    1 for line in difflib.unified_diff(
                        original_kernel.splitlines(), kernel_source.splitlines()
                    )
                    if line.startswith("+") and not line.startswith("+++")
                )

                new_kernel_result = {
                    "kernel_source": kernel_source,
                    "execution_result": {
                        "plan_id": plan_id,
                        "attempt": new_attempt,
                        "framework": framework,
                        "status": "success",
                        "correctness": "unverified",
                        "change_summary": change_summary,
                        "lines_changed": lines_changed,
                        "ready_for_profiling": True,
                    },
                }
                tool_messages.append(
                    ToolMessage(content="Kernel submitted successfully.", tool_call_id=tc_id)
                )

            elif new_attempt >= K:
                reason = (
                    "Submitted kernel is empty."
                    if not kernel_source.strip()
                    else f"No kernel marker found. Expected one of: {_KERNEL_MARKERS}"
                )
                new_kernel_result = {
                    "kernel_source": original_kernel,
                    "execution_result": {
                        "plan_id": plan_id,
                        "attempt": new_attempt,
                        "framework": "unknown",
                        "status": "implementation_failure",
                        "reason": reason,
                        "ready_for_profiling": False,
                    },
                }
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Validation failed (attempt {new_attempt}/{K}): {reason}. "
                            f"All attempts exhausted."
                        ),
                        tool_call_id=tc_id,
                    )
                )

            else:
                reason = (
                    "Submitted kernel is empty."
                    if not kernel_source.strip()
                    else f"No kernel marker found. Expected one of: {_KERNEL_MARKERS}"
                )
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Validation failed (attempt {new_attempt}/{K}): {reason}. "
                            f"Diagnose and resubmit a corrected version."
                        ),
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

    updates: dict = {"messages": tool_messages, "attempt": new_attempt}
    if new_kernel_result is not None:
        updates["kernel_result"] = new_kernel_result
    return updates


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_from_agent(state: ExecutorState) -> str:
    """Route to tools if the agent made tool calls, otherwise END (fallback)."""
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END
    return "tools"


def route_from_tools(state: ExecutorState) -> str:
    """Route to END if kernel_result is set, otherwise back to agent."""
    if state.get("kernel_result") is not None:
        return END
    return "agent"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def compile_executor_graph():
    """Return a compiled LangGraph sub-graph for one executor instance."""
    graph = StateGraph(ExecutorState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_from_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_from_tools, {"agent": "agent", END: END})

    return graph.compile()


executor_subgraph = compile_executor_graph()
