# Executor Agent Design

**Date**: 2026-04-02
**Status**: Approved

## Context

The executor agent was a POC implemented as a single Python function (`agents/executor.py`) with a
~120-line hardcoded system prompt containing all CUDA/Triton/Helion-specific transformation rules.
This is being replaced with a proper LangGraph sub-graph that mirrors the planner agent architecture,
giving the LLM full autonomy to reason from raw context rather than following baked-in rules.

---

## Goal

Replace `kernelopt/kernelopt/agents/executor.py` with a LangGraph ReAct sub-graph that:

1. Takes an optimization plan from the planner agent
2. Reads the kernel source and past memory via tools
3. Implements the change and submits via `submit_kernel`
4. Handles K-attempt retry internally (currently managed externally in `execute_plans`)
5. Has a thin system prompt — no hardcoded framework logic

---

## File Structure

```
kernelopt/kernelopt/agents/executor/
├── __init__.py          # exports run_executor_agent()
├── graph.py             # ExecutorState, agent_node, tool_node, routing, compile_executor_graph()
├── prompts.py           # SYSTEM_PROMPT + build_initial_message()
└── tools.py             # ExecutorToolServer + @tool wrappers + submit_kernel
```

The old `kernelopt/kernelopt/agents/executor.py` is deleted.
`langgraph_pipeline.py` imports `run_executor_agent` from the new package.

---

## State

```python
class ExecutorState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    kernel_result: dict | None   # set on success or K-exhaustion
    attempt: int                 # incremented by submit_kernel on each call
```

---

## Graph Topology

```
agent_node ──has tool_calls──→ tool_node
    ↑                               │
    └──── kernel_result not set ────┘
                                    │
                     kernel_result set (success or K exhausted)
                                    ↓
                                   END
```

**Routing:**
- `route_from_agent`: tool calls present → `"tools"`, else → `END`
- `route_from_tools`: `state["kernel_result"]` set → `END`, else → `"agent"`

---

## Tools

### `ExecutorToolServer`

Holds immutable state, thread-safe for concurrent reads. Never mutated after construction.

```python
class ExecutorToolServer:
    def __init__(self, kernel_source: str, optimization_plan: dict, memory: list[dict])
    def get_kernel_source(self) -> str
    def get_optimization_plan(self) -> dict
    def search_memory(self, query: str) -> list[dict]   # top-3 by keyword relevance
```

### `submit_kernel` — intercepted by `tool_node`

The `submit_kernel` `@tool` wrapper body is never called. `tool_node` intercepts it (identical
to how the planner intercepts `submit_plan`) and performs all logic directly:

1. Increment `state["attempt"]`
2. Validate submitted kernel: non-empty, contains at least one of `@triton.jit`, `__global__`, `@helion.kernel`, `@hl.kernel`
3. **Success**: write `kernel_result` to `state`, return success `ToolMessage`
4. **Failure + `state["attempt"] < K`**: return error `ToolMessage` (loop continues to agent)
5. **Failure + `state["attempt"] >= K`**: write `implementation_failure` to `state["kernel_result"]`, return final error `ToolMessage` (loop exits via `route_from_tools`)

### `execution_result` schema

```json
{
  "plan_id": "<from plan>",
  "attempt": 1,
  "framework": "<inferred from kernel markers>",
  "status": "success | implementation_failure",
  "correctness": "unverified",
  "change_summary": "<one sentence>",
  "lines_changed": "<int>",
  "ready_for_profiling": true
}
```

`framework` is inferred from kernel source markers — not hardcoded in the prompt.

### Tool list bound to LLM

```python
ALL_TOOLS = [get_kernel_source, get_optimization_plan, search_memory, submit_kernel]
```

`submit_kernel` is intercepted in `tool_node` (like `submit_plan` in the planner) to write to state.

---

## System Prompt & Initial Message

### System prompt (`prompts.py`) — thin, no framework rules

```
You are the Executor Agent in a GPU kernel optimization pipeline.

You receive an optimization plan and a kernel source file.
Your job is to implement exactly the change described in the plan.

Use your tools to understand the context:
  get_optimization_plan()     — read the plan you must implement
  get_kernel_source()         — read the current kernel source
  search_memory(query)        — check past experience for relevant patterns

When you have implemented the change, call submit_kernel(kernel_source, change_summary).
If submit_kernel returns an error, diagnose what went wrong and try again.
```

### `build_initial_message()`

```python
def build_initial_message(plan_direction: str, K: int) -> HumanMessage:
    return HumanMessage(content=(
        f"Implement the optimization plan.\n\n"
        f"Direction: {plan_direction}\n"
        f"You have up to {K} attempts via submit_kernel."
    ))
```

No framework rules, no correctness tolerances, no transformation recipes.

---

## Public Interface (`__init__.py`)

```python
def run_executor_agent(
    kernel_source: str,
    optimization_plan: dict,
    memory: list[dict],
    K: int,
    llm_config: dict,
) -> tuple[str, dict]:
    """
    Run the executor sub-graph.
    Returns (optimized_kernel_source, execution_result).
    Drop-in replacement for the old run_executor().
    """
```

---

## Changes to `langgraph_pipeline.py`

In `execute_plans`:

- **Remove** the `for attempt in range(1, cfg.K + 1)` retry loop — K is managed inside the graph
- **Remove** `previous_errors` accumulation
- **Add** `memory=state.get("optimization_memory", [])` passed to `run_executor_agent`
- **Replace** `run_executor(...)` with `run_executor_agent(...)`

The correctness gate, file saving, and profiling in `execute_plans` are unchanged.

---

## What is NOT changing

- `execute_plans` structure (correctness gate, file I/O, profiling)
- `PlanCandidate` and `GraphState` schemas
- The planner and profiler agents
- The `check_correctness` call (still external to the executor)
