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
high-level approach. Your plan MUST align with the strategy.

OPTIMIZATION GUIDANCE:
The kernel file may contain multiple GPU kernels (cuBLAS GEMM, cuDNN conv,
Triton @jit kernels, runtime kernels). Focus your plan on the @triton.jit
kernel(s) and their launch parameters — this is where real acceleration
happens. Library calls (extern_kernels.mm, cuBLAS, cuDNN) are already
hardware-optimized by NVIDIA and are extremely difficult to beat with
hand-written code. Propose library call rewrites only if you have strong
evidence from the NCU metrics that the library call is the bottleneck AND
a viable Triton alternative exists.

When you have gathered enough information, call submit_plan(plan) with
your plan. The plan must:
- Target a specific, measurable bottleneck in the @triton.jit kernel(s)
- Reference the exact construct in the kernel source to modify
- Be implementable in one focused diff by the Executor Agent
- Include enough detail in step.change and step.implementation_hints
  that the Executor does not need to re-read profiling data

Optimization strategies to consider (in priority order):
1. Add @triton.autotune with multiple configurations to explore tile sizes,
   num_warps, and num_stages automatically at runtime
2. Adjust tile sizes (XBLOCK, YBLOCK, num_warps, num_stages) for better occupancy
3. Improve memory access patterns (coalescing, vectorized loads, L2 compression)
4. Reduce register pressure or branch divergence
5. Change eviction policies and cache hints
6. Fuse adjacent @triton.jit pointwise kernels (NOT library call fusion)
7. Precision optimization (FP32 → BF16/FP16 where safe for stores/intermediates)
8. Algorithmic improvements (single-pass reductions, loop reordering, tiling)

For Helion kernels (@helion.kernel with hl.tile loops):
1. Adjust block_sizes in helion.Config for each hl.tile dimension
2. Change loop_orders to reorder nested hl.tile loops for memory locality
3. Set indexing='block_ptr' for TMA-based memory access
4. Set l2_grouping for L2 cache tile grouping
5. Adjust num_warps and num_stages in the Config
6. Remove config entirely to let Helion autotune (@helion.kernel())
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
