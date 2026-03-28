"""Planner agent: analyze profiling context + memory, produce an optimization plan."""

from __future__ import annotations

import json
import re
import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from kernelopt.llm import build_llm

PLANNER_SYSTEM_PROMPT = """\
You are a GPU kernel optimization expert. You reason step-by-step from profiling
data to a concrete, actionable optimization hypothesis — for any GPU framework.

## Reasoning Process (follow these 5 steps)

### Step 1 — Read the NCU rules first
The `profiling_context.rules` array contains NCU's own analysis. Read every OPT
rule's description and speedup_pct. Rank by speedup_pct descending (global first,
then local). The top-ranked rule is your first hypothesis candidate.

### Step 2 — Cross-reference with raw metrics
Dive into `profiling_context.sections` to understand *why* the top rule fired.
Find the specific metric values that NCU's rule description mentions.

### Step 3 — Look at the kernel source
Map the profiling signals back to the source code. Find the specific loop,
allocation, memory access pattern, or instruction sequence that corresponds
to the bottleneck.

Framework-specific reasoning:
- **CUDA**: Look for register pressure (large local arrays, many live variables),
  shared memory allocation size, global memory access patterns (stride, coalescing),
  loop structure and unroll opportunities.
- **Triton**: Look at tile sizes (BLOCK_M, BLOCK_N, BLOCK_K), tl.load/tl.store mask
  patterns, num_warps, num_stages. Triton auto-tunes many parameters — recommend
  which ones to change and in which direction.
- **Helion**: Look at tile hierarchy choices, memory access patterns in tile loops,
  and whether the kernel exploits available memory levels (L1, L2, HBM).
- **Unknown/Other**: Reason purely from profiling signals. Describe the transformation
  conceptually.

### Step 4 — Consult optimization memory (if provided)
Scan past experience items for patterns that match the current profiling signals.
Note: "Memory item X suggests Y — applicable here because Z."
Also note negative items to avoid repeating failed strategies.

### Step 5 — Form one hypothesis
Choose ONE optimization direction tied to specific metric values and specific
lines/constructs in the kernel source. It must be implementable in one diff.

## Output format

Your output MUST be a valid JSON object with this exact schema:
{
  "plan_id": "<short uuid>",
  "direction": "<descriptive snake_case name>",
  "framework": "<cuda|triton|helion|unknown>",
  "evidence": {
    "ncu_rule": "<Rule Name that motivated this, or null>",
    "ncu_rule_speedup_pct": <float or null>,
    "supporting_metrics": { "<Section>.<Metric Name>": <value>, ... },
    "reasoning": "<2-3 sentences: what the data says, why it points to this change>"
  },
  "step": {
    "action": "<reduce|increase|hoist|fuse|tile|pad|split|reorder|add|modify|remove|replace>",
    "target": "<which construct: loop, variable, allocation, parameter, instruction>",
    "change": "<specific change description>",
    "predicted_effect": "<which metric should improve, direction, rough magnitude>"
  },
  "correctness_constraints": ["<invariant the executor must preserve>"],
  "memory_item_used": "<item_id from optimization memory, or null>",
  "confidence": "<high|medium|low>",
  "confidence_reason": "<why: e.g. 'NCU rule directly points here' or 'inferring from indirect signal'>"
}

## Guidelines
- Output ONLY the JSON object, no markdown fences or extra text.
- Pick the single highest-impact optimization based on NCU rules sorted by speedup_pct.
- Be specific in the "change" field — reference exact code patterns.
- If optimization memory is provided, avoid repeating strategies marked as negative.

## Triton-specific valid optimizations
- Changing num_warps or num_stages in the kernel LAUNCH call (not kernel body)
- Adjusting BLOCK_SIZE (must remain power of 2)
- Using tl.math.exp2 instead of tl.exp (faster on GPU)
- Changing memory access patterns (pointer arithmetic, stride)
- Loop tiling/restructuring within the kernel
Do NOT suggest: manual shared memory, tl.shared_memory, tl.barrier, tl.syncthreads.

## CUDA-specific valid optimizations
- __launch_bounds__(block_size, min_blocks) for register pressure
- Shared memory tiling, vectorized loads (float4), loop unrolling
- Instruction fusion (rsqrtf, fmaf), coalesced memory access
- Grid/block dimension tuning

## Diversity requirement
When generating plan {plan_index} of N:
- Plan 0: target the top NCU rule by speedup_pct
- Plan 1+: target lower-ranked rules or orthogonal approaches
- Vary in scope (local tweak vs structural change) and risk level
- NEVER generate the same direction as a previous plan in this iteration
"""


def run_planner(
    profiling_context: dict,
    kernel_source: str,
    optimization_memory: list[dict],
    api_key: str | None = None,
    model: str = "Qwen/Qwen3-32B",
    base_url: str = "http://localhost:8000/v1",
    plan_index: int = 0,
    total_plans: int = 1,
    previous_directions: list[str] | None = None,
) -> dict:
    """Call the LLM to produce an optimization plan."""
    llm = build_llm(
        model=model,
        api_key=api_key,
        max_tokens=4096,
        temperature=0.3 + (plan_index * 0.1),  # increase temp for diversity
        base_url=base_url,
    )

    memory_section = ""
    if optimization_memory:
        memory_section = (
            "\n\n## Optimization Memory (past attempts)\n"
            f"```json\n{json.dumps(optimization_memory, indent=2)}\n```\n"
            "Avoid repeating strategies marked as 'negative' rewrite_type.\n"
            "Reference any applicable item_id in your memory_item_used field."
        )

    # Sort rules by speedup for emphasis
    rules = profiling_context.get("rules", [])
    sorted_rules = sorted(
        rules, key=lambda r: r.get("speedup_pct") or 0, reverse=True
    )

    diversity_section = ""
    if total_plans > 1:
        diversity_section = (
            f"\n\n## Diversity Constraint\n"
            f"This is plan {plan_index + 1} of {total_plans}.\n"
        )
        if plan_index == 0:
            diversity_section += "Target the TOP NCU rule by speedup_pct.\n"
        else:
            diversity_section += (
                f"Target a DIFFERENT rule or approach than the top one.\n"
                f"Focus on rule #{plan_index + 1} by speedup ranking.\n"
            )
        if previous_directions:
            diversity_section += (
                f"Do NOT use these directions (already taken): "
                f"{', '.join(previous_directions)}\n"
            )

    human_msg = (
        f"## Profiling Context\n"
        f"```json\n{json.dumps({**profiling_context, 'rules': sorted_rules}, indent=2)}\n```\n\n"
        f"## Kernel Source\n"
        f"```\n{kernel_source}\n```"
        f"{memory_section}"
        f"{diversity_section}"
    )

    response = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=human_msg),
    ])

    from kernelopt.llm import extract_json
    content = extract_json(response.content)

    plan = json.loads(content)

    if not plan.get("plan_id"):
        plan["plan_id"] = uuid.uuid4().hex[:8]
    if "memory_item_used" not in plan:
        plan["memory_item_used"] = None
    if "confidence_reason" not in plan:
        plan["confidence_reason"] = ""

    return plan
