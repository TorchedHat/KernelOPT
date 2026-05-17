# kernelopt/kernelopt/agents/executor/__init__.py
"""Executor Agent package.

Exports:
    run_executor_agent — runs one executor sub-graph instance
    executor_subgraph  — the compiled single-instance sub-graph (for testing/reuse)
"""

from __future__ import annotations

import traceback

from langchain_core.runnables import RunnableConfig

from kernelopt.agents.executor.graph import executor_subgraph
from kernelopt.agents.executor.prompts import build_initial_message
from kernelopt.agents.executor.tools import ExecutorToolServer


def run_executor_agent(
    kernel_source: str,
    optimization_plan: dict,
    memory: list[dict],
    K: int,
    llm_config: dict,
    llm_override=None,
    compile_error: str | None = None,
    correctness_error: str | None = None,
) -> tuple[str, dict]:
    """Run the executor sub-graph to apply one optimization plan.

    Args:
        kernel_source:      Current kernel source to be optimized.
        optimization_plan:  Plan dict from the planner agent.
        memory:             Optimization memory items from prior iterations.
        K:                  Maximum submit_kernel attempts allowed.
        llm_config:         Dict with keys model, api_key, project_id, region.
        llm_override:       Optional pre-built LLM (used in tests to avoid API calls).
        compile_error:      nvcc error from a previous compile attempt, or None.
        correctness_error:  Correctness failure message from a previous attempt, or None.
                            When set, the initial message includes the error so the
                            LLM can fix the numerical bug.

    Returns:
        (optimized_kernel_source, execution_result) — drop-in for old run_executor().
    """
    plan_direction = optimization_plan.get("direction", "unknown")
    plan_id = optimization_plan.get("plan_id", "")

    server = ExecutorToolServer(kernel_source, optimization_plan, memory)
    initial_msg = build_initial_message(
        plan_direction, K,
        compile_error=compile_error,
        correctness_error=correctness_error,
    )

    config = RunnableConfig(
        configurable={
            "tool_server": server,
            "llm_config": llm_config,
            "llm": llm_override,
            "K": K,
            "plan_id": plan_id,
            "original_kernel": kernel_source,
        }
    )

    try:
        result = executor_subgraph.invoke(
            {"messages": [initial_msg], "kernel_result": None, "attempt": 0},
            config=config,
        )
    except Exception as e:
        return kernel_source, {
            "plan_id": plan_id,
            "attempt": 0,
            "framework": "unknown",
            "status": "implementation_failure",
            "reason": str(e)[:500],
            "traceback": traceback.format_exc()[:1000],
            "ready_for_profiling": False,
        }

    kernel_result = result.get("kernel_result")
    if kernel_result is None:
        return kernel_source, {
            "plan_id": plan_id,
            "attempt": result.get("attempt", 0),
            "framework": "unknown",
            "status": "implementation_failure",
            "reason": "Agent did not call submit_kernel.",
            "ready_for_profiling": False,
        }

    return kernel_result["kernel_source"], kernel_result["execution_result"]


__all__ = ["run_executor_agent", "executor_subgraph"]
