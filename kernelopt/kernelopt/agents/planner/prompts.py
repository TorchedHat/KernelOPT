# kernelopt/kernelopt/agents/planner/prompts.py
"""System prompt and message builders for the Planner Agent."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage


SYSTEM_PROMPT = """\
You are the Planner Agent in a GPU kernel optimization pipeline.

You receive a profiling context from Nsight Compute (NCU) and the kernel
source code. Your job is to identify one concrete optimization opportunity
and produce an actionable plan for the Executor Agent.

Use your tools to investigate:
  query_profiling_context(aspect) — inspect specific profiling sections
    aspects: optimization_strategy, rules, occupancy, memory_workload,
             throughput, scheduler, launch, gpu_specs, instruction_stats,
             multi_kernel_summary
  get_kernel_source()             — read the kernel source code
  search_memory(query)           — check past optimization experience

IMPORTANT: Always check optimization_strategy first. It tells you the
high-level approach (fuse kernels, optimize individual kernel, or rewrite
and fuse). Your plan MUST align with the strategy. If the strategy says
"fuse", propose a plan that fuses operations. If it says "optimize_kernel",
propose hardware-level optimizations on the existing kernel.

When you have gathered enough information, call submit_plan(plan) with
your plan. The plan must:
- Target a specific, measurable bottleneck backed by metric values
- Reference the exact construct in the kernel source to modify
- Be implementable in one focused diff by the Executor Agent
- Include enough detail in step.change and step.implementation_hints
  that the Executor does not need to re-read profiling data

Optimization strategies to consider (in priority order):
1. Fuse adjacent kernels to eliminate intermediate memory read/write
2. Rewrite a library call (torch.mm, extern_kernels) as Triton for fusion
3. Improve memory access patterns (coalescing, L2 compression)
4. Adjust tile sizes (XBLOCK, YBLOCK) for better occupancy
5. Reduce register pressure or branch divergence
"""


def build_diversity_hint(plan_index: int, sorted_rules: list[dict]) -> str:
    """Return a per-instance diversity instruction for the initial human message.

    sorted_rules must be pre-sorted by speedup_pct descending.
    """
    if not sorted_rules:
        return (
            "No NCU rules available — reason from raw metrics. "
            "Focus on the most significant bottleneck visible in the sections."
        )

    if plan_index == 0:
        rule = sorted_rules[0]
        speedup = rule.get("speedup_pct")
        speedup_str = f" (estimated speedup: {speedup}%)" if speedup else ""
        return f"Target the top NCU rule: '{rule['name']}'{speedup_str}."

    if plan_index < len(sorted_rules):
        rule = sorted_rules[plan_index]
        speedup = rule.get("speedup_pct")
        speedup_str = f" (estimated speedup: {speedup}%)" if speedup else ""
        return (
            f"Target rule '{rule['name']}'{speedup_str}. "
            f"This must be a different direction from plan 0."
        )

    return (
        "No more NCU rules to target. Reason from raw metrics orthogonal "
        "to the directions already covered by other plans in this iteration."
    )


def build_initial_message(
    diversity_hint: str,
    all_tried_directions: list[str],
) -> HumanMessage:
    """Build the opening human message for a planner sub-graph instance."""
    directions_block = (
        json.dumps(all_tried_directions, indent=2)
        if all_tried_directions
        else "(none yet)"
    )
    content = (
        f"Analyze the kernel profiling context and produce one optimization plan.\n\n"
        f"Diversity instruction: {diversity_hint}\n\n"
        f"Do NOT propose any direction from this list "
        f"(already tried in prior iterations):\n{directions_block}"
    )
    return HumanMessage(content=content)
