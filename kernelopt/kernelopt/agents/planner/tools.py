# kernelopt/kernelopt/agents/planner/tools.py
"""PlannerToolServer and LangChain tool definitions for the Planner Agent."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool


class PlannerToolServer:
    """Holds immutable profiling data and serves tool calls for planner sub-graphs.

    Thread-safe for concurrent reads — no mutation after construction.
    Callers must not mutate the dicts returned by query_profiling_context
    or search_memory; they share memory with internal state.
    """

    def __init__(
        self,
        profiling_context: dict,
        kernel_source: str,
        memory: list[dict],
    ) -> None:
        self.profiling_context = profiling_context
        self.kernel_source = kernel_source
        self.memory = memory

    def query_profiling_context(self, aspect: str) -> dict:
        """Return a slice of the profiling context for the given aspect."""
        ctx = self.profiling_context
        sections = ctx.get("sections", {})
        launch = ctx.get("launch", {})

        if aspect == "rules":
            rules = ctx.get("rules", [])
            return {
                "rules": sorted(
                    rules,
                    key=lambda r: r.get("speedup_pct") if r.get("speedup_pct") is not None else 0,
                    reverse=True,
                )
            }

        if aspect == "occupancy":
            return {
                "sections": {"Occupancy": sections.get("Occupancy", {})},
                "launch": {
                    "waves_per_sm": launch.get("waves_per_sm"),
                    "registers_per_thread": launch.get("registers_per_thread"),
                    "block_size": launch.get("block_size"),
                },
            }

        if aspect == "memory_workload":
            return {
                "sections": {
                    "Memory Workload Analysis": sections.get(
                        "Memory Workload Analysis", {}
                    )
                }
            }

        if aspect == "throughput":
            return {
                "sections": {
                    "GPU Speed Of Light Throughput": sections.get(
                        "GPU Speed Of Light Throughput", {}
                    )
                }
            }

        if aspect == "scheduler":
            return {
                "sections": {
                    "Scheduler Statistics": sections.get("Scheduler Statistics", {})
                }
            }

        if aspect == "launch":
            return {"launch": launch}

        if aspect == "gpu_specs":
            return {"gpu_specs": ctx.get("gpu_specs", {})}

        if aspect == "instruction_stats":
            return {
                "sections": {
                    "Instruction Statistics": sections.get(
                        "Instruction Statistics", {}
                    )
                }
            }

        if aspect == "multi_kernel_summary":
            summary = ctx.get("multi_kernel_summary", "")
            if summary:
                return {"multi_kernel_summary": summary}
            return {"multi_kernel_summary": "Single kernel file (no multi-kernel info)."}

        if aspect == "optimization_strategy":
            strategy = ctx.get("optimization_strategy", {})
            if strategy:
                return {"optimization_strategy": strategy}
            return {"optimization_strategy": {"strategy_type": "optimize_kernel"}}

        valid = (
            "rules, occupancy, memory_workload, throughput, "
            "scheduler, launch, gpu_specs, instruction_stats, "
            "multi_kernel_summary, optimization_strategy"
        )
        raise ValueError(f"Unknown aspect: {aspect!r}. Must be one of: {valid}")

    def get_kernel_source(self) -> str:
        """Return the full kernel source."""
        return self.kernel_source

    def search_memory(self, query: str) -> list[dict]:
        """Return up to 3 memory items most relevant to the query.

        Word-level matching: "register" matches items containing "register"
        or "registers" as a word. Words shorter than 3 characters are
        filtered out to avoid trivial matches (e.g., "a" matching everything).
        An empty query returns the first 3 items in list order.
        """
        if not self.memory:
            return []

        # Filter short words that produce noise matches
        query_words = [w for w in query.lower().split() if len(w) >= 3]
        if not query_words:
            return self.memory[:3]

        def score(item: dict) -> int:
            text = " ".join([
                item.get("direction", ""),
                item.get("strategy_title", ""),
                item.get("strategy_description", ""),
                item.get("applicable_when", ""),
            ]).lower()
            text_words = text.split()
            # Word-level matching: query word must appear within a text word
            return sum(
                1 for w in query_words
                if any(w in tw for tw in text_words)
            )

        return sorted(self.memory, key=score, reverse=True)[:3]


# ---------------------------------------------------------------------------
# LangChain @tool wrappers — delegate to PlannerToolServer via configurable
# ---------------------------------------------------------------------------


@tool
def query_profiling_context(
    aspect: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> dict:
    """Query a slice of the NCU profiling context.

    aspect must be one of:
      rules            — NCU optimization rules sorted by estimated speedup
      occupancy        — Occupancy section + key launch parameters
      memory_workload  — Memory Workload Analysis section
      throughput       — GPU Speed of Light throughput percentages
      scheduler        — Scheduler Statistics (stall reasons)
      launch           — Full launch configuration (block size, grid, waves/SM, registers)
      gpu_specs        — GPU hardware specs (peak bandwidth, TFLOPS, SM count)
      instruction_stats — Instruction Statistics (IPC)
    """
    server: PlannerToolServer = config["configurable"]["tool_server"]
    return server.query_profiling_context(aspect)


@tool
def get_kernel_source(
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Return the full kernel source code.

    Call this when you need to map a profiling signal to a specific construct
    in the code — a loop, tile parameter, allocation, or instruction.
    """
    server: PlannerToolServer = config["configurable"]["tool_server"]
    return server.get_kernel_source()


@tool
def search_memory(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> list[dict]:
    """Search past optimization experience for relevant strategies.

    query — natural language, e.g. "register pressure triton" or "tile size occupancy".
    Returns up to 3 memory items ranked by keyword relevance.
    Each item includes: direction, strategy_description, applicable_when,
    do_not_apply_when, rewrite_type (positive/negative/implementation_flaw), speedup.
    """
    server: PlannerToolServer = config["configurable"]["tool_server"]
    return server.search_memory(query)


@tool
def submit_plan(plan: dict) -> str:
    """Submit your final optimization plan. Call this when you have gathered
    enough information from profiling context and kernel source.

    The plan dict must include:
      plan_id             — short unique identifier (e.g. "plan_001")
      direction           — snake_case name (e.g. "reduce_register_pressure")
      framework           — "cuda" | "triton" | "helion" | "unknown"
      evidence            — {ncu_rule, ncu_rule_speedup_pct, supporting_metrics, reasoning}
      step                — {action, target, change, predicted_effect,
                             implementation_hints, success_criteria}
      correctness_constraints — list of strings
      architecture_notes  — string or null
      memory_item_used    — item_id string or null
      confidence          — "high" | "medium" | "low"
      confidence_reason   — string

    step.change must describe EXACTLY what the executor must modify (old → new values).
    step.implementation_hints must give concrete before/after code snippets.
    """
    # This function body is never called — tool_node intercepts submit_plan
    # and handles validation + state update directly.
    return "received"


# Map of tool name → tool function for tool_node dispatch.
# submit_plan is intentionally excluded — handled separately in tool_node.
_TOOL_MAP: dict[str, Any] = {
    "query_profiling_context": query_profiling_context,
    "get_kernel_source": get_kernel_source,
    "search_memory": search_memory,
}

# All tools to bind to the LLM (including submit_plan).
ALL_TOOLS = [query_profiling_context, get_kernel_source, search_memory, submit_plan]
