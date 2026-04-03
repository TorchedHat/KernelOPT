# kernelopt/kernelopt/agents/executor/tools.py
"""ExecutorToolServer and LangChain tool definitions for the Executor Agent."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool


class ExecutorToolServer:
    """Holds immutable kernel/plan/memory data and serves tool calls for the executor sub-graph.

    Thread-safe for concurrent reads — no mutation after construction.
    """

    def __init__(
        self,
        kernel_source: str,
        optimization_plan: dict,
        memory: list[dict],
    ) -> None:
        self.kernel_source = kernel_source
        self.optimization_plan = optimization_plan
        self.memory = memory

    def get_kernel_source(self) -> str:
        """Return the full kernel source to be optimized."""
        return self.kernel_source

    def get_optimization_plan(self) -> dict:
        """Return the optimization plan the executor must implement."""
        return self.optimization_plan

    def search_memory(self, query: str) -> list[dict]:
        """Return up to 3 memory items most relevant to the query.

        Keyword matching is substring-based. Empty query returns first 3 items.
        """
        if not self.memory:
            return []

        query_words = set(query.lower().split())

        def score(item: dict) -> int:
            text = " ".join([
                item.get("direction", ""),
                item.get("strategy_title", ""),
                item.get("strategy_description", ""),
                item.get("applicable_when", ""),
            ]).lower()
            return sum(1 for w in query_words if w in text)

        return sorted(self.memory, key=score, reverse=True)[:3]


# ---------------------------------------------------------------------------
# LangChain @tool wrappers — delegate to ExecutorToolServer via configurable
# ---------------------------------------------------------------------------

@tool
def get_kernel_source(
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Return the full kernel source code that needs to be optimized.

    Call this to read the current kernel before making changes.
    """
    server: ExecutorToolServer = config["configurable"]["tool_server"]
    return server.get_kernel_source()


@tool
def get_optimization_plan(
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> dict:
    """Return the optimization plan to implement.

    Contains: plan_id, direction, framework, step.change, step.target,
    step.predicted_effect, correctness_constraints, evidence.
    """
    server: ExecutorToolServer = config["configurable"]["tool_server"]
    return server.get_optimization_plan()


@tool
def search_memory(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> list[dict]:
    """Search past optimization experience for relevant patterns.

    query — natural language, e.g. "register pressure triton" or "tile size failure".
    Returns up to 3 memory items ranked by keyword relevance.
    Each item includes: direction, strategy_description, applicable_when,
    do_not_apply_when, rewrite_type (positive/negative/implementation_flaw), speedup.
    """
    server: ExecutorToolServer = config["configurable"]["tool_server"]
    return server.search_memory(query)


@tool
def submit_kernel(kernel_source: str, change_summary: str) -> str:
    """Submit the complete optimized kernel source.

    kernel_source  — the full file with ONLY the plan's change applied.
    change_summary — one sentence describing exactly what changed.

    Call this when you have implemented the optimization. If validation fails,
    diagnose the error in the response and resubmit a corrected version.
    """
    # Body never called — tool_node intercepts submit_kernel and handles
    # validation + state update directly (same pattern as submit_plan in planner).
    return "received"


# Map of tool name → tool function for tool_node dispatch.
# submit_kernel is intentionally excluded — handled separately in tool_node.
_TOOL_MAP: dict[str, Any] = {
    "get_kernel_source": get_kernel_source,
    "get_optimization_plan": get_optimization_plan,
    "search_memory": search_memory,
}

# All tools to bind to the LLM (including submit_kernel).
ALL_TOOLS = [get_kernel_source, get_optimization_plan, search_memory, submit_kernel]
