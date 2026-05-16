# kernelopt/kernelopt/agents/planner/__init__.py
"""Planner Agent package.

Exports:
    run_planner_agent — orchestrates N parallel planner sub-graph instances
    planner_subgraph  — the compiled single-instance sub-graph (for testing/reuse)
"""

from __future__ import annotations

import string
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.runnables import RunnableConfig

from kernelopt.agents.planner.graph import planner_subgraph
from kernelopt.agents.planner.prompts import build_diversity_hint, build_initial_message
from kernelopt.agents.planner.tools import PlannerToolServer
from kernelopt.llm import DEFAULT_MODEL

PLAN_LETTERS = list(string.ascii_uppercase)


def run_planner_agent(
    profiling_context: dict,
    kernel_source: str,
    memory: list[dict],
    N: int,
    llm_config: dict,
    all_tried_directions: list[str],
    llm_override=None,
) -> list[tuple[str, dict]]:
    """Run N planner sub-graph instances concurrently and return collected plans.

    Args:
        profiling_context: Output from the profiler agent.
        kernel_source: Current best kernel source code.
        memory: Optimization memory items from prior iterations.
        N: Number of plans to generate.
        llm_config: Dict with keys model, api_key, project_id, region.
        all_tried_directions: Directions tried in all prior iterations (for cross-iteration diversity).
        llm_override: Optional pre-built LLM to inject (used in tests to avoid API calls).

    Returns:
        List of (plan_letter, plan_dict) tuples, deduplicated by direction.
    """
    server = PlannerToolServer(profiling_context, kernel_source, memory)

    sorted_rules = sorted(
        profiling_context.get("rules", []),
        key=lambda r: r.get("speedup_pct") if r.get("speedup_pct") is not None else 0,
        reverse=True,
    )

    strategy = profiling_context.get("optimization_strategy", {})
    strategy_type = strategy.get("strategy_type", "optimize_kernel")
    plan_guidance = strategy.get("plan_guidance", [])

    def run_one(i: int) -> tuple[str, dict] | None:
        strategy_prefix = ""
        if strategy_type in ("fuse", "rewrite_and_fuse"):
            strategy_prefix = (
                f"OVERALL STRATEGY: {strategy_type} — {strategy.get('reasoning', '')} "
                f"Bottleneck: {strategy.get('primary_target', 'unknown')}. "
            )

        if strategy_type in ("fuse", "rewrite_and_fuse") and i == 0:
            hint = (
                f"{strategy_prefix}"
                f"You MUST propose a plan that fuses or rewrites kernels into Triton. "
                f"Fusion opportunity: {strategy.get('fusion_opportunity', 'see multi_kernel_summary')}."
            )
        elif plan_guidance and i < len(plan_guidance):
            hint = f"{strategy_prefix}{plan_guidance[i]}"
        else:
            ncu_hint = build_diversity_hint(i, sorted_rules)
            hint = f"{strategy_prefix}{ncu_hint}" if strategy_prefix else ncu_hint
        initial_msg = build_initial_message(hint, all_tried_directions)

        config = RunnableConfig(
            configurable={
                "tool_server": server,
                "llm_config": llm_config,
                "llm": llm_override,  # None in production, mock in tests
            }
        )

        try:
            result = planner_subgraph.invoke(
                {"messages": [initial_msg], "plan": None},
                config=config,
            )
            plan = result.get("plan")
            if plan is None:
                return None
            return (PLAN_LETTERS[i], plan)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=N) as pool:
        futures = [pool.submit(run_one, i) for i in range(N)]
        raw = [f.result() for f in as_completed(futures)]

    # Restore letter order (as_completed returns in completion order)
    ordered = sorted([p for p in raw if p], key=lambda p: p[0])

    # Deduplicate by direction (keep first occurrence in letter order)
    seen: set[str] = set()
    deduped: list[tuple[str, dict]] = []
    for letter, plan in ordered:
        direction = plan.get("direction", "")
        if direction not in seen:
            seen.add(direction)
            deduped.append((letter, plan))

    return deduped


def run_planner(
    profiling_context: dict,
    kernel_source: str,
    optimization_memory: list[dict],
    api_key: str | None = None,
    model: str | None = None,
    project_id: str | None = None,
    region: str | None = None,
    plan_index: int = 0,
    total_plans: int = 1,
    previous_directions: list[str] | None = None,
) -> dict:
    """
    Legacy API for single-plan generation (tests and older call sites).

    This is a compatibility wrapper. New code should use run_planner_agent().

    Args:
        plan_index: Ignored (kept for backwards compatibility)
        total_plans: Ignored (kept for backwards compatibility)

    Returns: A single plan dict (not a list of tuples).
    """
    llm_config = {
        "model": model or DEFAULT_MODEL,
        "api_key": api_key,
        "project_id": project_id or "",
        "region": region or "us-east5",
    }

    all_tried_directions = previous_directions or []

    # Call new API with N=1 to get a single plan
    plans = run_planner_agent(
        profiling_context=profiling_context,
        kernel_source=kernel_source,
        memory=optimization_memory,
        N=1,
        llm_config=llm_config,
        all_tried_directions=all_tried_directions,
        llm_override=None,
    )

    if not plans:
        raise RuntimeError("Planner failed to generate a plan")

    # Return just the plan dict (strip the letter)
    _, plan = plans[0]
    return plan


__all__ = ["run_planner_agent", "run_planner", "planner_subgraph"]
