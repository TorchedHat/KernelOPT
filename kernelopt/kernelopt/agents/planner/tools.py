# kernelopt/kernelopt/agents/planner/tools.py
"""PlannerToolServer and LangChain tool definitions for the Planner Agent."""

from __future__ import annotations


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

        valid = (
            "rules, occupancy, memory_workload, throughput, "
            "scheduler, launch, gpu_specs, instruction_stats"
        )
        raise ValueError(f"Unknown aspect: {aspect!r}. Must be one of: {valid}")

    def get_kernel_source(self) -> str:
        """Return the full kernel source."""
        return self.kernel_source

    def search_memory(self, query: str) -> list[dict]:
        """Return up to 3 memory items most relevant to the query.

        Keyword matching is substring-based: "register" matches "registers_per_thread".
        An empty query returns the first 3 items in list order.
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
