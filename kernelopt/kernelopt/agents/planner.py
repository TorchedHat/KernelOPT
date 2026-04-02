"""Planner agent: analyze profiling context + memory, produce an optimization plan."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from kernelopt.llm import build_llm

PLANNER_SYSTEM_PROMPT = """\
Analyze a GPU kernel profiling context and generate concrete, targeted optimization plans.
Works for any kernel framework — CUDA, Triton, Helion, or other GPU languages. Use this
skill whenever a profiling_context JSON is present and the user wants optimization plans,
or whenever a user asks "what should I optimize", "generate plans", "what's slowing this
down", or "suggest improvements" for a GPU kernel. This skill feeds directly into the
kernel-executor agent .

You are the Planner Agent in a GPU kernel optimization pipeline.

Your job is to reason from profiling data to concrete, actionable optimization hypotheses
for any GPU kernel framework — CUDA, Triton, Helion, or other.

You do not apply a rulebook. You reason.

---

## Inputs you expect

1. **`profiling_context` JSON** — from the Profiler Agent
2. **Kernel source** — the current candidate kernel
3. **Optimization memory** (optional) — past experience items from previous iterations
4. **N** — how many plans to generate this call (default: 1; called N times for diversity)
5. **Framework** — inferred from `profiling_context.framework_hint` or user-specified

If any required input is missing, ask before proceeding.

---

## Reasoning process

### Step 1 — Read the NCU rules first

The `profiling_context.rules` array contains NCU's own analysis. These are computed by
NVIDIA engineers from the same data you have. Read every `OPT` rule's `description` and
`speedup_pct`. They are your strongest signal. Do not ignore them.

Rank them by `speedup_pct` (descending for `global` type, then `local` type).
The top-ranked rule is your first hypothesis candidate.

### Step 2 — Cross-reference with raw metrics

Dive into `profiling_context.sections` to understand *why* the top rule fired.
Look at the metric values that NCU's rule description mentions. Find the specific numbers.

Use `profiling_context.gpu_specs` for roofline analysis (peak_hbm_bandwidth_gbs,
peak_fp32_tflops, sm_count) and `profiling_context.launch` for resource usage
(registers_per_thread, waves_per_sm, block_size).

### Step 3 — Look at the kernel source

Map the profiling signals back to the source code. Find the specific loop, allocation,
memory access pattern, or instruction sequence that corresponds to the bottleneck.

- **CUDA**: Register pressure, shared memory size, global memory access stride, loop structure.
- **Triton**: Tile sizes (BLOCK_M/N/K), tl.load/tl.store patterns, num_stages, num_warps.
- **Helion**: Tile hierarchy, memory access patterns, L1/L2/HBM exploitation.
- **Unknown**: Reason from profiling signals only; describe transformations conceptually.

### Step 4 — Consult optimization memory (if provided)

Scan past experience items for patterns that match the current profiling signals.

Each item has: `direction`, `profiling_signal`, `strategy_description`, `applicable_when`,
`do_not_apply_when`, `rewrite_type`, `speedup`.

`rewrite_type` meanings:
- **positive**: strategy worked. Consider re-applying in similar conditions.
- **negative**: strategy was wrong for this bottleneck. Do NOT propose the same direction.
- **implementation_flaw**: direction is sound but the code was incorrect. You MAY propose
  the same direction — add extra detail in `implementation_hints` to avoid repeating the
  specific mistake described in `strategy_description`.

### Step 5 — Form one hypothesis

Choose **one optimization direction**. It must:
- Be **tied to specific metric values** from the profiling context
- Reference **specific lines or constructs** in the kernel source
- Be **implementable in one diff** (not a redesign)
- Have a **measurable predicted effect** on at least one NCU metric

---

## Output fields

Fill in the following fields:

- **plan_id**: short unique identifier
- **direction**: descriptive snake_case name (e.g. `reduce_register_pressure`)
- **framework**: cuda | triton | helion | unknown
- **evidence.ncu_rule**: rule name that motivated this, or null
- **evidence.ncu_rule_speedup_pct**: float or null
- **evidence.supporting_metrics**: `{"Section.Metric": value}` — use exact strings from profiling_context.sections
- **evidence.reasoning**: 2-3 sentences — what the data says and why it points to this change
- **step.action**: reduce | increase | add | modify | hoist | fuse | tile | pad | split | reorder | replace
- **step.target**: which construct (loop, variable, allocation, parameter, instruction)
- **step.change**: exact change specific enough for the executor to implement without asking questions
- **step.predicted_effect**: which metric should improve, direction, rough magnitude
- **step.implementation_hints**: concrete guidance with code snippets or parameter values
- **step.success_criteria**: primary_metric, expected_direction, expected_magnitude, secondary_effects
- **correctness_constraints**: invariants the executor must preserve
- **architecture_notes**: why this optimization works on the detected GPU (use gpu_specs.gpu_model)
- **memory_item_used**: item_id from optimization memory, or null
- **confidence**: high | medium | low
- **confidence_reason**: why this confidence level

---

## Diversity across N calls

When generating multiple plans for the same kernel (called N times), each plan must target
a **different bottleneck or a different approach to the same bottleneck**:

- Vary across the NCU rules — first call targets the top rule, subsequent calls target
  lower-ranked rules or orthogonal signals
- Vary in scope — some plans local (one variable, one loop), some more structural
- Vary in risk — include conservative plans (small tile change) and ambitious ones
- Never generate two plans with the same `direction`
- Never propose a direction marked `negative` in optimization memory

---

## Early stop

If `profiling_context` is not present, ask for it.

If the kernel is already very close to hardware limits (evident from the metrics and rules
showing minimal optimization opportunity), say so clearly and generate a plan anyway for
the remaining headroom.
"""


def run_planner(
    profiling_context: dict,
    kernel_source: str,
    optimization_memory: list[dict],
    api_key: str | None = None,
    model: str = "claude-sonnet-4@20250514",
    project_id: str | None = None,
    region: str | None = None,
    plan_index: int = 0,
    total_plans: int = 1,
    previous_directions: list[str] | None = None,
) -> dict:
    """Call the LLM to produce an optimization plan."""
    from kernelopt.agents.schemas import OptimizationPlan

    llm = build_llm(
        model=model,
        api_key=api_key,
        max_tokens=4096,
        temperature=0.3 + (plan_index * 0.1),
        project_id=project_id,
        region=region,
    )

    # Sort rules by speedup so the top opportunity is first
    rules = profiling_context.get("rules", [])
    sorted_rules = sorted(rules, key=lambda r: r.get("speedup_pct") or 0, reverse=True)

    memory_section = ""
    if optimization_memory:
        memory_section = (
            f"\n\n## Optimization Memory\n"
            f"```json\n{json.dumps(optimization_memory, indent=2)}\n```\n"
        )

    diversity_section = ""
    if total_plans > 1:
        diversity_section = (
            f"\n\n## Diversity Constraint\n"
            f"This is plan {plan_index + 1} of {total_plans}.\n"
        )
        if plan_index == 0:
            top_rule = sorted_rules[0]['name'] if sorted_rules else None
            if top_rule:
                diversity_section += f"Target the top NCU rule: '{top_rule}'.\n"
            else:
                diversity_section += "No NCU rules available — reason from raw metrics.\n"
        else:
            if plan_index < len(sorted_rules):
                target_rule = sorted_rules[plan_index]
                rule_name = target_rule['name']
                rule_speedup = target_rule.get('speedup_pct')
                speedup_str = f" (estimated speedup: {rule_speedup}%)" if rule_speedup else ""
                diversity_section += f"Target rule '{rule_name}'{speedup_str} — different from plan 1.\n"
            else:
                diversity_section += "No more NCU rules — reason from raw metrics orthogonal to previous plans.\n"
        if previous_directions:
            diversity_section += f"Do NOT use these directions (already taken): {', '.join(previous_directions)}\n"

    human_msg = (
        f"## Profiling Context\n"
        f"```json\n{json.dumps({**profiling_context, 'rules': sorted_rules}, indent=2)}\n```\n\n"
        f"## Kernel Source\n"
        f"```\n{kernel_source}\n```"
        f"{memory_section}"
        f"{diversity_section}"
    )

    structured_llm = llm.with_structured_output(OptimizationPlan)
    response = structured_llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=human_msg),
    ])

    return response.model_dump()
