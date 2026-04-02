# Planner Agent Design

**Date:** 2026-04-02
**Status:** Approved

---

## Overview

Replace the current `planner.py` — which hardcodes a wall-of-text reasoning algorithm in a Python string — with a proper tool-calling agent. The planner becomes a LangGraph sub-graph that investigates profiling data and kernel source via tools, then emits a structured `OptimizationPlan` by calling `submit_plan`. N plans are generated concurrently via N parallel sub-graph instances sharing a single `PlannerToolServer`.

---

## Inputs

| Input | Source | How delivered |
|---|---|---|
| `profiling_context` | Profiler agent output | Held in `PlannerToolServer`, queried via tool |
| `kernel_source` | Current best kernel | Held in `PlannerToolServer`, queried via tool |
| `optimization_memory` | Summarizer / prior iterations | Held in `PlannerToolServer`, queried via tool |
| `diversity_hint` | Outer pipeline (per-instance) | Injected into initial human message |
| `previous_directions` | Accumulated within iteration | Injected into initial human message |

---

## Output

One `OptimizationPlan` (Pydantic schema in `kernelopt/agents/schemas.py`) per sub-graph instance. The outer `generate_plans` node collects N plans and returns `{"plans": [(letter, plan), ...]}`.

The plan is specific enough that the executor does not need to re-read profiling data — `step.target`, `step.change`, and `step.implementation_hints` tell the executor exactly what to touch and where in the kernel source.

---

## Architecture

```
generate_plans (outer LangGraph node)
│
├── PlannerToolServer(profiling_context, kernel_source, memory)
│
├── ThreadPoolExecutor — N concurrent runners
│   ├── planner_subgraph(plan_index=0, hint="top NCU rule")
│   ├── planner_subgraph(plan_index=1, hint="2nd NCU rule")
│   └── planner_subgraph(plan_index=N-1, hint="...")
│
└── collect N OptimizationPlan results → {"plans": [...]}
```

### Sub-graph topology

```
[agent_node] ──has tool calls──► [tool_node] ──► back to [agent_node]
      │
      └── submit_plan called → [END]
      └── no tool calls      → [END]  (fallback)
```

**State:**
```python
class PlannerState(TypedDict):
    messages: list[BaseMessage]
    plan: dict | None   # populated when submit_plan is called
```

**Routing:**
```python
def route(state) -> str:
    last = state["messages"][-1]
    if not last.tool_calls:
        return END
    if any(tc["name"] == "submit_plan" for tc in last.tool_calls):
        return END
    return "tool_node"
```

The `tool_node` intercepts `submit_plan`, validates its argument against `OptimizationPlan`, writes to `state["plan"]`, and appends a `ToolMessage` — but does not execute any side effect. All other tools are executed normally.

---

## Tools

All tools are read-only. Served by `PlannerToolServer`, injected via LangGraph's `configurable` context.

### `query_profiling_context(aspect: str) -> dict`

Returns a slice of the profiling context. `aspect` must be one of:

| aspect | Returns |
|---|---|
| `"rules"` | Full `rules` array, sorted by `speedup_pct` descending |
| `"occupancy"` | `sections["Occupancy"]` + `launch.waves_per_sm` + `launch.registers_per_thread` |
| `"memory_workload"` | `sections["Memory Workload Analysis"]` |
| `"throughput"` | `sections["GPU Speed Of Light Throughput"]` |
| `"scheduler"` | `sections["Scheduler Statistics"]` |
| `"launch"` | Full `launch` dict |
| `"gpu_specs"` | Full `gpu_specs` dict |
| `"instruction_stats"` | `sections["Instruction Statistics"]` |

The agent decides which aspects to query. It does not receive the full context upfront.

### `get_kernel_source() -> str`

Returns the full kernel source. Called when the agent needs to map a profiling signal (e.g. high register count) to a specific construct in the code (e.g. a loop variable, tile parameter, shared memory allocation).

### `search_memory(query: str) -> list[dict]`

Accepts a natural language query. Performs keyword matching against `direction`, `strategy_title`, `strategy_description`, and `applicable_when` fields of memory items. Returns top-3 matches. The agent evaluates relevance.

### `submit_plan(plan: dict) -> None`

Terminal tool. The agent calls this when it has gathered enough information. The `plan` dict is validated against `OptimizationPlan` on receipt. Calling this exits the agent loop.

---

## System Prompt

Minimal — role and output contract only. No hardcoded reasoning steps.

```
You are the Planner Agent in a GPU kernel optimization pipeline.

You receive a profiling context from Nsight Compute (NCU) and the kernel
source code. Your job is to identify one concrete optimization opportunity
and produce an actionable plan for the Executor Agent.

Use your tools to investigate:
- query_profiling_context(aspect) — inspect specific profiling sections
- get_kernel_source()             — read the kernel source code
- search_memory(query)           — check past optimization experience

When you have gathered enough information, call submit_plan(plan) with
your plan. The plan must:
- Target a specific, measurable bottleneck backed by metric values
- Reference the exact construct in the kernel source to modify
- Be implementable in one focused diff by the Executor Agent
- Include enough detail in step.change and step.implementation_hints
  that the Executor does not need to re-read profiling data
```

The `diversity_hint` and `all_tried_directions` are injected per-instance into the **initial human message**, not the system prompt. The system prompt is identical across all N instances.

### `build_diversity_hint(i, sorted_rules) -> str`

Produces a per-instance instruction injected into the initial human message:
- `i=0` → `"Target the top NCU rule: '<name>' (estimated speedup: X%)"`
- `i=1..len(rules)-1` → `"Target rule '<name>' (estimated speedup: X%) — different from plan 0"`
- `i >= len(rules)` → `"No more NCU rules — reason from raw metrics orthogonal to previous plans"`

### `build_initial_message(hint, all_tried_directions) -> HumanMessage`

Wraps the diversity hint and cross-iteration history into the opening human turn:

```
Analyze the kernel profiling context and produce one optimization plan.

Diversity instruction: {hint}

Do NOT propose any direction from this list (already tried in prior iterations):
{all_tried_directions as JSON}
```

### Intra-iteration diversity

Because all N sub-graphs run **concurrently**, `previous_directions` cannot be accumulated within a single iteration. Intra-iteration diversity is enforced entirely by `build_diversity_hint` assigning each instance a different NCU rule to target. Post-generation, duplicate `direction` values are deduplicated by keeping the first occurrence.

Cross-iteration diversity (never repeat a previously tried direction) is enforced via `all_tried_directions`, built from `state["iteration_results"]` before launching the pool.

---

## Parallel Execution

```python
def generate_plans(state: GraphState) -> dict:
    server = PlannerToolServer(
        profiling_context=state["current_profiling_context"],
        kernel_source=state["best_kernel"],
        memory=state.get("optimization_memory", []),
    )

    sorted_rules = sorted(
        server.profiling_context.get("rules", []),
        key=lambda r: r.get("speedup_pct") or 0,
        reverse=True,
    )

    all_tried_directions = [
        c.direction
        for result in state.get("iteration_results", [])
        for c in result.candidates
    ]

    def run_one(i: int) -> tuple[str, dict] | None:
        hint = build_diversity_hint(i, sorted_rules)
        msg = build_initial_message(hint, all_tried_directions)
        result = planner_subgraph.invoke(
            {"messages": [msg], "plan": None},
            config={"configurable": {"tool_server": server}},
        )
        return (PLAN_LETTERS[i], result["plan"]) if result["plan"] else None

    with ThreadPoolExecutor(max_workers=N) as pool:
        futures = [pool.submit(run_one, i) for i in range(N)]
        plans = [f.result() for f in futures]

    # Deduplicate by direction (keep first occurrence)
    seen: set[str] = set()
    deduped = []
    for p in plans:
        if p and p[1].get("direction") not in seen:
            seen.add(p[1]["direction"])
            deduped.append(p)

    return {"plans": deduped}
```

`PlannerToolServer` holds immutable data — concurrent read access is safe with no locks.

---

## File Layout

```
kernelopt/agents/
├── planner/
│   ├── __init__.py     # exports run_planner_agent(), planner_subgraph
│   ├── graph.py        # sub-graph: agent_node, tool_node, route, compile()
│   ├── tools.py        # PlannerToolServer + 4 tool definitions
│   └── prompts.py      # SYSTEM_PROMPT + build_initial_message()
├── planner.py          # deleted (replaced by planner/ package)
└── schemas.py          # unchanged — OptimizationPlan lives here
```

`langgraph_pipeline.py` imports `run_planner_agent` from `kernelopt.agents.planner` and calls it inside the `generate_plans` node, replacing the current `run_planner` loop.

---

## What Does Not Change

- `OptimizationPlan` schema (`schemas.py`) — unchanged
- The executor, profiler, summarizer agents — unchanged
- The outer LangGraph pipeline graph topology — unchanged
- The `generate_plans` node signature (inputs/outputs) — unchanged
- Memory format — unchanged
