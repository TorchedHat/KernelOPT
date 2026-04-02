# Planner Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded-system-prompt planner with a proper tool-calling LangGraph sub-graph where N plans are generated concurrently, each agent investigating profiling data and kernel source via tools before submitting a plan.

**Architecture:** N parallel `planner_subgraph` instances (one per plan) share a single `PlannerToolServer` holding immutable profiling context, kernel source, and memory. Each sub-graph runs a ReAct loop (agent → tools → agent) and terminates when the LLM calls `submit_plan(plan)`. The outer `generate_plans` LangGraph node launches all N instances via `ThreadPoolExecutor` and collects results.

**Tech Stack:** LangGraph `StateGraph`, LangChain tool calling (`@tool`, `InjectedToolArg`, `RunnableConfig`), Pydantic v2 (`OptimizationPlan`), `concurrent.futures.ThreadPoolExecutor`, pytest.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| CREATE | `kernelopt/kernelopt/agents/planner/__init__.py` | Exports `run_planner_agent()`, `planner_subgraph` |
| CREATE | `kernelopt/kernelopt/agents/planner/tools.py` | `PlannerToolServer` class + 4 LangChain `@tool` functions |
| CREATE | `kernelopt/kernelopt/agents/planner/prompts.py` | `SYSTEM_PROMPT`, `build_diversity_hint()`, `build_initial_message()` |
| CREATE | `kernelopt/kernelopt/agents/planner/graph.py` | `PlannerState`, `agent_node`, `tool_node`, routing, `compile_planner_graph()` |
| MODIFY | `kernelopt/kernelopt/langgraph_pipeline.py` | Replace `generate_plans` node to call `run_planner_agent()` |
| DELETE | `kernelopt/kernelopt/agents/planner.py` | Replaced by package |
| CREATE | `kernelopt/tests/__init__.py` | Empty — marks tests as package |
| CREATE | `kernelopt/tests/agents/__init__.py` | Empty |
| CREATE | `kernelopt/tests/conftest.py` | Shared fixtures: sample profiling context, kernel source, memory |
| CREATE | `kernelopt/tests/agents/test_planner_tools.py` | Unit tests for `PlannerToolServer` methods |
| CREATE | `kernelopt/tests/agents/test_planner_graph.py` | Unit tests for routing, `tool_node`, sub-graph integration |

---

## Task 1: Test infrastructure and fixtures

**Files:**
- Create: `kernelopt/tests/__init__.py`
- Create: `kernelopt/tests/agents/__init__.py`
- Create: `kernelopt/tests/conftest.py`

- [ ] **Step 1: Create empty package markers**

```bash
touch kernelopt/tests/__init__.py kernelopt/tests/agents/__init__.py
```

- [ ] **Step 2: Write conftest.py with shared fixtures**

```python
# kernelopt/tests/conftest.py
import pytest


@pytest.fixture
def sample_profiling_context():
    return {
        "kernel_name": "triton_softmax",
        "kernel_name_full": "triton__0d1d2d3_softmax_kernel",
        "kernel_id": 0,
        "framework_hint": "triton",
        "device": "NVIDIA H100 SXM5",
        "compute_capability": "9.0",
        "launch": {
            "block_size": 1024,
            "grid_size": 4096,
            "sm_count": 132,
            "waves_per_sm": 0.62,
            "registers_per_thread": 48,
            "dynamic_smem_per_block_bytes": 0,
            "static_smem_per_block_bytes": 0,
            "threads": 1048576,
        },
        "gpu_specs": {
            "gpu_model": "H100 SXM5",
            "gpu_model_detected": True,
            "peak_hbm_bandwidth_gbs": 3350.0,
            "peak_fp32_tflops": 67.0,
            "sm_count": 132,
        },
        "sections": {
            "GPU Speed Of Light Throughput": {
                "Compute (SM) Throughput": {"value": 18.4, "unit": "%"},
                "Memory Throughput": {"value": 72.1, "unit": "%"},
            },
            "Occupancy": {
                "Theoretical Occupancy": {"value": 62.5, "unit": "%"},
                "Achieved Occupancy": {"value": 50.5, "unit": "%"},
                "Block Limit Registers": {"value": 5, "unit": "block"},
                "Block Limit Warps": {"value": 8, "unit": "block"},
            },
            "Memory Workload Analysis": {
                "DRAM Throughput": {"value": 68.3, "unit": "%"},
            },
            "Scheduler Statistics": {
                "No Eligible": {"value": 41.2, "unit": "%"},
            },
            "Instruction Statistics": {
                "IPC Active": {"value": 1.2, "unit": "inst/cycle"},
            },
        },
        "rules": [
            {
                "name": "TheoreticalOccupancy",
                "type": "OPT",
                "description": "Register usage limits occupancy to 62.5%.",
                "speedup_type": "global",
                "speedup_pct": 24.0,
            },
            {
                "name": "MemoryBound",
                "type": "INF",
                "description": "Kernel is memory bound.",
                "speedup_type": "local",
                "speedup_pct": 8.0,
            },
        ],
    }


@pytest.fixture
def sample_kernel_source():
    return """\
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x = tl.load(x_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE))
    y = tl.softmax(x, axis=0)
    tl.store(y_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE), y)
"""


@pytest.fixture
def sample_memory():
    return [
        {
            "item_id": "abc123",
            "iteration": 1,
            "speedup": 1.28,
            "rewrite_type": "positive",
            "framework": "triton",
            "direction": "reduce_tile_size_for_occupancy",
            "strategy_title": "Halve tile size to reduce register pressure",
            "strategy_description": (
                "When Block Limit Registers is the binding constraint and "
                "registers_per_thread > 32, reducing tile size lowers register usage."
            ),
            "applicable_when": (
                "Block Limit Registers < Block Limit Warps AND registers_per_thread > 32"
            ),
            "do_not_apply_when": "Kernel is compute-bound (high SM throughput).",
            "profiling_signal": {},
            "slow_pseudocode": "",
            "fast_pseudocode": "",
            "framework_notes": None,
        },
        {
            "item_id": "def456",
            "iteration": 2,
            "speedup": 0.91,
            "rewrite_type": "negative",
            "framework": "triton",
            "direction": "increase_num_warps",
            "strategy_title": "Increasing num_warps caused regression",
            "strategy_description": "More warps increased register pressure further.",
            "applicable_when": "N/A",
            "do_not_apply_when": "When register pressure is already high.",
            "profiling_signal": {},
            "slow_pseudocode": "",
            "fast_pseudocode": "",
            "framework_notes": None,
        },
    ]
```

- [ ] **Step 3: Verify pytest discovers the fixtures**

```bash
cd kernelopt && python -m pytest tests/ --collect-only 2>&1 | head -20
```

Expected: no import errors, 0 tests collected (no test files yet).

- [ ] **Step 4: Commit**

```bash
git add kernelopt/tests/
git commit -m "test: add test infrastructure and shared fixtures for planner agent"
```

---

## Task 2: PlannerToolServer (tools.py — server class)

**Files:**
- Create: `kernelopt/kernelopt/agents/planner/tools.py`
- Create: `kernelopt/tests/agents/test_planner_tools.py`

- [ ] **Step 1: Write failing tests for PlannerToolServer**

```python
# kernelopt/tests/agents/test_planner_tools.py
import pytest
from kernelopt.agents.planner.tools import PlannerToolServer


def test_query_rules_sorted_by_speedup(sample_profiling_context, sample_memory, sample_kernel_source):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, sample_memory)
    result = server.query_profiling_context("rules")
    rules = result["rules"]
    assert len(rules) == 2
    # TheoreticalOccupancy (24.0%) should come before MemoryBound (8.0%)
    assert rules[0]["name"] == "TheoreticalOccupancy"
    assert rules[1]["name"] == "MemoryBound"


def test_query_occupancy_includes_launch_fields(sample_profiling_context, sample_kernel_source):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, [])
    result = server.query_profiling_context("occupancy")
    assert "Occupancy" in result["sections"]
    assert result["launch"]["waves_per_sm"] == 0.62
    assert result["launch"]["registers_per_thread"] == 48
    assert result["launch"]["block_size"] == 1024


def test_query_memory_workload(sample_profiling_context, sample_kernel_source):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, [])
    result = server.query_profiling_context("memory_workload")
    assert "Memory Workload Analysis" in result["sections"]
    assert result["sections"]["Memory Workload Analysis"]["DRAM Throughput"]["value"] == 68.3


def test_query_throughput(sample_profiling_context, sample_kernel_source):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, [])
    result = server.query_profiling_context("throughput")
    assert "GPU Speed Of Light Throughput" in result["sections"]


def test_query_scheduler(sample_profiling_context, sample_kernel_source):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, [])
    result = server.query_profiling_context("scheduler")
    assert "Scheduler Statistics" in result["sections"]


def test_query_launch(sample_profiling_context, sample_kernel_source):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, [])
    result = server.query_profiling_context("launch")
    assert result["launch"]["block_size"] == 1024
    assert result["launch"]["sm_count"] == 132


def test_query_gpu_specs(sample_profiling_context, sample_kernel_source):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, [])
    result = server.query_profiling_context("gpu_specs")
    assert result["gpu_specs"]["gpu_model"] == "H100 SXM5"


def test_query_instruction_stats(sample_profiling_context, sample_kernel_source):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, [])
    result = server.query_profiling_context("instruction_stats")
    assert "Instruction Statistics" in result["sections"]


def test_query_unknown_aspect_raises(sample_profiling_context, sample_kernel_source):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, [])
    with pytest.raises(ValueError, match="Unknown aspect"):
        server.query_profiling_context("garbage")


def test_get_kernel_source(sample_profiling_context, sample_kernel_source):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, [])
    assert server.get_kernel_source() == sample_kernel_source


def test_search_memory_ranks_by_keyword(sample_profiling_context, sample_kernel_source, sample_memory):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, sample_memory)
    results = server.search_memory("register pressure occupancy")
    assert len(results) <= 3
    # "reduce_tile_size_for_occupancy" mentions both "register" and "occupancy"
    assert results[0]["direction"] == "reduce_tile_size_for_occupancy"


def test_search_memory_empty(sample_profiling_context, sample_kernel_source):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, [])
    results = server.search_memory("anything")
    assert results == []


def test_search_memory_returns_at_most_three(sample_profiling_context, sample_kernel_source):
    memory = [
        {
            "item_id": str(i), "direction": f"dir_{i}",
            "strategy_title": "foo bar", "strategy_description": "foo bar",
            "applicable_when": "foo bar", "do_not_apply_when": "",
            "framework": "triton", "rewrite_type": "positive",
            "speedup": 1.1, "iteration": 1, "profiling_signal": {},
            "slow_pseudocode": "", "fast_pseudocode": "", "framework_notes": None,
        }
        for i in range(5)
    ]
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, memory)
    results = server.search_memory("foo bar")
    assert len(results) <= 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd kernelopt && python -m pytest tests/agents/test_planner_tools.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'kernelopt.agents.planner'`

- [ ] **Step 3: Create the planner package directory**

```bash
mkdir -p kernelopt/kernelopt/agents/planner
touch kernelopt/kernelopt/agents/planner/__init__.py
```

- [ ] **Step 4: Implement PlannerToolServer in tools.py**

```python
# kernelopt/kernelopt/agents/planner/tools.py
"""PlannerToolServer and LangChain tool definitions for the Planner Agent."""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from kernelopt.agents.schemas import OptimizationPlan


class PlannerToolServer:
    """Holds immutable profiling data and serves tool calls for planner sub-graphs.

    Thread-safe for concurrent reads — no mutation after construction.
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
                    key=lambda r: r.get("speedup_pct") or 0,
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
        """Return up to 3 memory items most relevant to the query."""
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd kernelopt && python -m pytest tests/agents/test_planner_tools.py -v
```

Expected: all 13 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add kernelopt/kernelopt/agents/planner/ kernelopt/tests/agents/test_planner_tools.py
git commit -m "feat: add PlannerToolServer with profiling context slicing and memory search"
```

---

## Task 3: LangChain tool wrappers (tools.py — @tool functions)

**Files:**
- Modify: `kernelopt/kernelopt/agents/planner/tools.py` (append)
- Modify: `kernelopt/tests/agents/test_planner_tools.py` (append)

- [ ] **Step 1: Write failing tests for the tool wrappers**

Append to `kernelopt/tests/agents/test_planner_tools.py`:

```python
# --- Tool wrapper tests ---

from langchain_core.runnables import RunnableConfig
from kernelopt.agents.planner.tools import (
    query_profiling_context as tool_query,
    get_kernel_source as tool_get_source,
    search_memory as tool_search,
    submit_plan as tool_submit,
    _TOOL_MAP,
)


def _make_config(server: PlannerToolServer) -> RunnableConfig:
    return RunnableConfig(configurable={"tool_server": server})


def test_tool_query_profiling_context(sample_profiling_context, sample_kernel_source):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, [])
    config = _make_config(server)
    result = tool_query.invoke({"aspect": "rules"}, config=config)
    assert "rules" in result
    assert result["rules"][0]["name"] == "TheoreticalOccupancy"


def test_tool_get_kernel_source(sample_profiling_context, sample_kernel_source):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, [])
    config = _make_config(server)
    result = tool_get_source.invoke({}, config=config)
    assert "softmax_kernel" in result


def test_tool_search_memory(sample_profiling_context, sample_kernel_source, sample_memory):
    server = PlannerToolServer(sample_profiling_context, sample_kernel_source, sample_memory)
    config = _make_config(server)
    result = tool_search.invoke({"query": "register occupancy"}, config=config)
    assert isinstance(result, list)
    assert result[0]["direction"] == "reduce_tile_size_for_occupancy"


def test_tool_map_contains_read_tools():
    assert "query_profiling_context" in _TOOL_MAP
    assert "get_kernel_source" in _TOOL_MAP
    assert "search_memory" in _TOOL_MAP
    # submit_plan is handled specially in tool_node — not in _TOOL_MAP
    assert "submit_plan" not in _TOOL_MAP


def test_submit_plan_tool_has_name():
    assert tool_submit.name == "submit_plan"
```

- [ ] **Step 2: Run tests to see them fail**

```bash
cd kernelopt && python -m pytest tests/agents/test_planner_tools.py -v -k "tool_" 2>&1 | head -20
```

Expected: `ImportError` — tool functions not yet defined.

- [ ] **Step 3: Append tool wrappers to tools.py**

Add after the `PlannerToolServer` class definition:

```python
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
_TOOL_MAP: dict[str, object] = {
    "query_profiling_context": query_profiling_context,
    "get_kernel_source": get_kernel_source,
    "search_memory": search_memory,
}

# All tools to bind to the LLM (including submit_plan).
ALL_TOOLS = [query_profiling_context, get_kernel_source, search_memory, submit_plan]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd kernelopt && python -m pytest tests/agents/test_planner_tools.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kernelopt/kernelopt/agents/planner/tools.py kernelopt/tests/agents/test_planner_tools.py
git commit -m "feat: add LangChain tool wrappers for planner agent (query, get_source, search, submit)"
```

---

## Task 4: Prompts (prompts.py)

**Files:**
- Create: `kernelopt/kernelopt/agents/planner/prompts.py`
- Create: `kernelopt/tests/agents/test_planner_prompts.py`

- [ ] **Step 1: Write failing tests**

```python
# kernelopt/tests/agents/test_planner_prompts.py
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from kernelopt.agents.planner.prompts import (
    SYSTEM_PROMPT,
    build_diversity_hint,
    build_initial_message,
)


SAMPLE_RULES = [
    {"name": "TheoreticalOccupancy", "speedup_pct": 24.0},
    {"name": "MemoryBound", "speedup_pct": 8.0},
]


def test_system_prompt_contains_role():
    assert "Planner Agent" in SYSTEM_PROMPT
    assert "submit_plan" in SYSTEM_PROMPT


def test_system_prompt_has_no_step_by_step_algorithm():
    # Must NOT contain hardcoded reasoning steps
    assert "Step 1" not in SYSTEM_PROMPT
    assert "Step 2" not in SYSTEM_PROMPT
    assert "Step 3" not in SYSTEM_PROMPT


def test_build_diversity_hint_first_plan():
    hint = build_diversity_hint(0, SAMPLE_RULES)
    assert "TheoreticalOccupancy" in hint
    assert "24.0" in hint


def test_build_diversity_hint_second_plan():
    hint = build_diversity_hint(1, SAMPLE_RULES)
    assert "MemoryBound" in hint
    assert "8.0" in hint


def test_build_diversity_hint_beyond_rules():
    hint = build_diversity_hint(5, SAMPLE_RULES)
    assert "raw metrics" in hint.lower()


def test_build_diversity_hint_empty_rules():
    hint = build_diversity_hint(0, [])
    assert "raw metrics" in hint.lower() or "no" in hint.lower()


def test_build_initial_message_is_human_message():
    hint = build_diversity_hint(0, SAMPLE_RULES)
    msg = build_initial_message(hint, [])
    assert isinstance(msg, HumanMessage)


def test_build_initial_message_contains_hint():
    hint = "Target rule 'TheoreticalOccupancy' (estimated speedup: 24.0%)"
    msg = build_initial_message(hint, [])
    assert "TheoreticalOccupancy" in msg.content


def test_build_initial_message_contains_tried_directions():
    msg = build_initial_message("hint", ["reduce_register_pressure", "increase_tile_k"])
    assert "reduce_register_pressure" in msg.content
    assert "increase_tile_k" in msg.content


def test_build_initial_message_empty_directions():
    msg = build_initial_message("hint", [])
    # Should not crash; content should still be a non-empty string
    assert isinstance(msg.content, str)
    assert len(msg.content) > 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd kernelopt && python -m pytest tests/agents/test_planner_prompts.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'kernelopt.agents.planner.prompts'`

- [ ] **Step 3: Implement prompts.py**

```python
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
  get_kernel_source()             — read the kernel source code
  search_memory(query)           — check past optimization experience

When you have gathered enough information, call submit_plan(plan) with
your plan. The plan must:
- Target a specific, measurable bottleneck backed by metric values
- Reference the exact construct in the kernel source to modify
- Be implementable in one focused diff by the Executor Agent
- Include enough detail in step.change and step.implementation_hints
  that the Executor does not need to re-read profiling data
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd kernelopt && python -m pytest tests/agents/test_planner_prompts.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kernelopt/kernelopt/agents/planner/prompts.py kernelopt/tests/agents/test_planner_prompts.py
git commit -m "feat: add planner agent system prompt and message builders"
```

---

## Task 5: Sub-graph (graph.py)

**Files:**
- Create: `kernelopt/kernelopt/agents/planner/graph.py`
- Create: `kernelopt/tests/agents/test_planner_graph.py`

- [ ] **Step 1: Write failing tests for routing and tool_node**

```python
# kernelopt/tests/agents/test_planner_graph.py
import json
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from kernelopt.agents.planner.graph import route_from_agent, route_from_tools, tool_node
from kernelopt.agents.planner.tools import PlannerToolServer


VALID_PLAN = {
    "plan_id": "p001",
    "direction": "reduce_register_pressure",
    "framework": "triton",
    "evidence": {
        "ncu_rule": "TheoreticalOccupancy",
        "ncu_rule_speedup_pct": 24.0,
        "supporting_metrics": {"Occupancy.Block Limit Registers": 5},
        "reasoning": "Register usage is the binding occupancy constraint.",
    },
    "step": {
        "action": "reduce",
        "target": "BLOCK_SIZE parameter",
        "change": "Reduce BLOCK_SIZE from 1024 to 512 in the triton.Config",
        "predicted_effect": "Registers per thread should drop from 48 to ~24, occupancy improves.",
        "implementation_hints": [
            "Change BLOCK_SIZE=1024 to BLOCK_SIZE=512 in the @triton.autotune config."
        ],
        "success_criteria": {
            "primary_metric": "Occupancy.Theoretical Occupancy",
            "expected_direction": "increase",
            "expected_magnitude": "20-30%",
            "secondary_effects": "Duration may decrease if occupancy was the bottleneck.",
        },
    },
    "correctness_constraints": ["Output values must match reference softmax to atol=1e-3"],
    "architecture_notes": "H100 SXM5 has 65536 registers per SM; fewer regs per thread = more blocks.",
    "memory_item_used": None,
    "confidence": "high",
    "confidence_reason": "NCU rule directly points to register-limited occupancy.",
}


# --- route_from_agent ---

def test_route_from_agent_no_tool_calls_goes_to_end():
    msg = AIMessage(content="I am done.")
    state = {"messages": [HumanMessage(content="go"), msg], "plan": None}
    assert route_from_agent(state) == END


def test_route_from_agent_with_tool_calls_goes_to_tools():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "query_profiling_context", "args": {"aspect": "rules"}, "id": "tc1"}],
    )
    state = {"messages": [HumanMessage(content="go"), msg], "plan": None}
    assert route_from_agent(state) == "tools"


def test_route_from_agent_empty_tool_calls_goes_to_end():
    msg = AIMessage(content="Done.", tool_calls=[])
    state = {"messages": [HumanMessage(content="go"), msg], "plan": None}
    assert route_from_agent(state) == END


# --- route_from_tools ---

def test_route_from_tools_plan_set_goes_to_end():
    state = {"messages": [], "plan": {"direction": "something"}}
    assert route_from_tools(state) == END


def test_route_from_tools_no_plan_goes_to_agent():
    state = {"messages": [], "plan": None}
    assert route_from_tools(state) == "agent"


# --- tool_node ---

@pytest.fixture
def server(sample_profiling_context, sample_kernel_source, sample_memory):
    return PlannerToolServer(sample_profiling_context, sample_kernel_source, sample_memory)


def _config(server):
    return RunnableConfig(configurable={"tool_server": server})


def test_tool_node_executes_query_tool(server):
    msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "query_profiling_context",
            "args": {"aspect": "rules"},
            "id": "tc1",
        }],
    )
    state = {"messages": [HumanMessage(content="go"), msg], "plan": None}
    result = tool_node(state, _config(server))
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ToolMessage)
    assert result.get("plan") is None  # not set — not a submit_plan call


def test_tool_node_executes_get_kernel_source(server):
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "get_kernel_source", "args": {}, "id": "tc2"}],
    )
    state = {"messages": [HumanMessage(content="go"), msg], "plan": None}
    result = tool_node(state, _config(server))
    tool_msg = result["messages"][0]
    assert "softmax_kernel" in tool_msg.content


def test_tool_node_submit_plan_sets_state_plan(server):
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "submit_plan", "args": {"plan": VALID_PLAN}, "id": "tc3"}],
    )
    state = {"messages": [HumanMessage(content="go"), msg], "plan": None}
    result = tool_node(state, _config(server))
    assert result["plan"] is not None
    assert result["plan"]["direction"] == "reduce_register_pressure"
    # ToolMessage should acknowledge submission
    assert isinstance(result["messages"][0], ToolMessage)


def test_tool_node_submit_plan_invalid_schema_does_not_set_plan(server):
    bad_plan = {"plan_id": "x"}  # missing required fields
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "submit_plan", "args": {"plan": bad_plan}, "id": "tc4"}],
    )
    state = {"messages": [HumanMessage(content="go"), msg], "plan": None}
    result = tool_node(state, _config(server))
    assert result.get("plan") is None
    # ToolMessage should contain validation error
    assert "validation" in result["messages"][0].content.lower() or \
           "failed" in result["messages"][0].content.lower() or \
           "error" in result["messages"][0].content.lower()
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd kernelopt && python -m pytest tests/agents/test_planner_graph.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'kernelopt.agents.planner.graph'`

- [ ] **Step 3: Implement graph.py**

```python
# kernelopt/kernelopt/agents/planner/graph.py
"""Planner Agent LangGraph sub-graph definition."""

from __future__ import annotations

import json

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from kernelopt.agents.schemas import OptimizationPlan
from kernelopt.agents.planner.prompts import SYSTEM_PROMPT
from kernelopt.agents.planner.tools import ALL_TOOLS, _TOOL_MAP


class PlannerState(TypedDict, total=False):
    messages: list
    plan: dict | None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def agent_node(state: PlannerState, config: RunnableConfig) -> dict:
    """Call the LLM with tools bound. Accepts a pre-built LLM via configurable for testing."""
    from langchain_core.messages import SystemMessage

    from kernelopt.llm import build_llm

    llm = config["configurable"].get("llm")
    if llm is None:
        llm_cfg = config["configurable"]["llm_config"]
        llm = build_llm(**llm_cfg)

    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: PlannerState, config: RunnableConfig) -> dict:
    """Execute tool calls from the last agent message.

    submit_plan is intercepted: validates the plan dict against OptimizationPlan
    and writes it to state["plan"]. All other tools are executed normally.
    """
    last = state["messages"][-1]
    tool_messages: list[ToolMessage] = []
    new_plan: dict | None = None

    for tc in last.tool_calls:
        name = tc["name"]
        args = tc["args"]
        tc_id = tc["id"]

        if name == "submit_plan":
            try:
                plan = OptimizationPlan(**args["plan"])
                new_plan = plan.model_dump()
                tool_messages.append(
                    ToolMessage(content="Plan submitted successfully.", tool_call_id=tc_id)
                )
            except Exception as exc:
                tool_messages.append(
                    ToolMessage(
                        content=f"Validation failed: {exc}. Fix the plan and resubmit.",
                        tool_call_id=tc_id,
                    )
                )

        else:
            tool_fn = _TOOL_MAP.get(name)
            if tool_fn is None:
                tool_messages.append(
                    ToolMessage(content=f"Unknown tool: {name!r}", tool_call_id=tc_id)
                )
                continue
            try:
                result = tool_fn.invoke(args, config=config)
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(result, default=str),
                        tool_call_id=tc_id,
                    )
                )
            except Exception as exc:
                tool_messages.append(
                    ToolMessage(content=f"Tool error: {exc}", tool_call_id=tc_id)
                )

    updates: dict = {"messages": tool_messages}
    if new_plan is not None:
        updates["plan"] = new_plan
    return updates


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_from_agent(state: PlannerState) -> str:
    """Route to tools if agent made tool calls, otherwise END (fallback)."""
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END
    return "tools"


def route_from_tools(state: PlannerState) -> str:
    """Route to END if plan was submitted, otherwise back to agent."""
    if state.get("plan") is not None:
        return END
    return "agent"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def compile_planner_graph():
    """Return a compiled LangGraph sub-graph for one planner instance."""
    graph = StateGraph(PlannerState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_from_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_from_tools, {"agent": "agent", END: END})

    return graph.compile()


planner_subgraph = compile_planner_graph()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd kernelopt && python -m pytest tests/agents/test_planner_graph.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kernelopt/kernelopt/agents/planner/graph.py kernelopt/tests/agents/test_planner_graph.py
git commit -m "feat: add planner agent sub-graph with ReAct loop and submit_plan termination"
```

---

## Task 6: Package init and run_planner_agent()

**Files:**
- Modify: `kernelopt/kernelopt/agents/planner/__init__.py`

- [ ] **Step 1: Write failing test for run_planner_agent**

Append to `kernelopt/tests/agents/test_planner_graph.py`:

```python
# --- run_planner_agent integration ---

from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from kernelopt.agents.planner import run_planner_agent
from kernelopt.agents.planner.tools import PlannerToolServer


def _make_mock_llm(tool_calls_sequence):
    """Return a mock LLM that emits AIMessages with the given tool_calls in sequence."""
    mock_llm = MagicMock()
    responses = [
        AIMessage(content="", tool_calls=tcs) for tcs in tool_calls_sequence
    ]
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = responses
    return mock_llm


def test_run_planner_agent_returns_plans(
    sample_profiling_context, sample_kernel_source, sample_memory
):
    """run_planner_agent with N=1 should return 1 (letter, plan) tuple."""
    mock_llm = _make_mock_llm([
        # First call: submit immediately
        [{"name": "submit_plan", "args": {"plan": VALID_PLAN}, "id": "tc1"}],
    ])

    plans = run_planner_agent(
        profiling_context=sample_profiling_context,
        kernel_source=sample_kernel_source,
        memory=sample_memory,
        N=1,
        llm_config={"model": "unused", "api_key": None, "project_id": None, "region": None},
        all_tried_directions=[],
        llm_override=mock_llm,
    )
    assert len(plans) == 1
    letter, plan = plans[0]
    assert letter == "A"
    assert plan["direction"] == "reduce_register_pressure"


def test_run_planner_agent_deduplicates_directions(
    sample_profiling_context, sample_kernel_source, sample_memory
):
    """Two plans with the same direction should be deduplicated to one."""
    mock_llm = _make_mock_llm([
        [{"name": "submit_plan", "args": {"plan": VALID_PLAN}, "id": "tc1"}],
        [{"name": "submit_plan", "args": {"plan": VALID_PLAN}, "id": "tc2"}],
    ])

    plans = run_planner_agent(
        profiling_context=sample_profiling_context,
        kernel_source=sample_kernel_source,
        memory=sample_memory,
        N=2,
        llm_config={"model": "unused", "api_key": None, "project_id": None, "region": None},
        all_tried_directions=[],
        llm_override=mock_llm,
    )
    # Both plans have direction "reduce_register_pressure" → deduplicated to 1
    assert len(plans) == 1


def test_run_planner_agent_skips_failed_instances(
    sample_profiling_context, sample_kernel_source, sample_memory
):
    """An instance that never calls submit_plan returns None and is excluded."""
    mock_llm = _make_mock_llm([
        # No tool calls → agent gives up → plan is None
        [],
    ])

    plans = run_planner_agent(
        profiling_context=sample_profiling_context,
        kernel_source=sample_kernel_source,
        memory=sample_memory,
        N=1,
        llm_config={"model": "unused", "api_key": None, "project_id": None, "region": None},
        all_tried_directions=[],
        llm_override=mock_llm,
    )
    assert plans == []
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd kernelopt && python -m pytest tests/agents/test_planner_graph.py -v -k "run_planner_agent" 2>&1 | head -20
```

Expected: `ImportError` — `run_planner_agent` not yet defined.

- [ ] **Step 3: Implement __init__.py**

```python
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
        key=lambda r: r.get("speedup_pct") or 0,
        reverse=True,
    )

    def run_one(i: int) -> tuple[str, dict] | None:
        hint = build_diversity_hint(i, sorted_rules)
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


__all__ = ["run_planner_agent", "planner_subgraph"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd kernelopt && python -m pytest tests/agents/test_planner_graph.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kernelopt/kernelopt/agents/planner/__init__.py
git commit -m "feat: add run_planner_agent with parallel sub-graph execution and direction deduplication"
```

---

## Task 7: Wire into langgraph_pipeline.py and delete old planner.py

**Files:**
- Modify: `kernelopt/kernelopt/langgraph_pipeline.py` (lines 107–167, `generate_plans` function)
- Delete: `kernelopt/kernelopt/agents/planner.py`

- [ ] **Step 1: Replace generate_plans in langgraph_pipeline.py**

Replace the entire `generate_plans` function (lines 107–167) with:

```python
def generate_plans(state: GraphState) -> dict:
    """Generate N diverse optimization plans concurrently via parallel planner sub-graphs."""
    from kernelopt.agents.planner import run_planner_agent

    cfg = state["config"]
    profiling_ctx = state["current_profiling_context"]
    iteration = state.get("iteration", 0)
    baseline = state.get("baseline_latency_ns", float("inf"))
    best_speedup = round(baseline / state["best_latency_ns"], 2)

    console.print(
        f"\n[bold]Iteration {iteration + 1}/{cfg.T}[/]  "
        f"Best: [green]{best_speedup}x[/]  "
        f"Memory: {len(state.get('optimization_memory', []))} items"
    )
    console.print(f"  Generating {cfg.N} optimization plans...")

    # Build cross-iteration direction history
    all_tried_directions = [
        c.direction
        for result in state.get("iteration_results", [])
        for c in result.candidates
    ]

    llm_config = {
        "model": cfg.planner_model or cfg.model,
        "api_key": cfg.api_key,
        "project_id": cfg.project_id,
        "region": cfg.region,
    }

    plans = run_planner_agent(
        profiling_context=profiling_ctx,
        kernel_source=state["best_kernel"],
        memory=state.get("optimization_memory", []),
        N=cfg.N,
        llm_config=llm_config,
        all_tried_directions=all_tried_directions,
    )

    for letter, plan in plans:
        direction = plan.get("direction", "unknown")
        confidence = plan.get("confidence", "?")
        console.print(
            f"    Plan {letter}: {direction} [dim](confidence: {confidence})[/]"
        )

    return {"plans": plans}
```

Also remove the now-unused import at the top of `langgraph_pipeline.py`:

```python
# Remove this line:
from kernelopt.agents.planner import run_planner
```

- [ ] **Step 2: Delete the old planner.py**

```bash
rm kernelopt/kernelopt/agents/planner.py
```

- [ ] **Step 3: Verify the pipeline imports without errors**

```bash
cd kernelopt && python -c "from kernelopt.langgraph_pipeline import build_graph; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Run all tests to verify nothing is broken**

```bash
cd kernelopt && python -m pytest tests/ -v
```

Expected: all tests PASS. No import errors.

- [ ] **Step 5: Commit**

```bash
git add kernelopt/kernelopt/langgraph_pipeline.py
git rm kernelopt/kernelopt/agents/planner.py
git commit -m "feat: wire run_planner_agent into pipeline, remove old hardcoded planner"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Covered by |
|---|---|
| Minimal system prompt (role + output contract, no hardcoded steps) | Task 4 — `SYSTEM_PROMPT`, tested for absence of "Step 1/2/3" |
| Tool-calling agent with coarse tools | Task 3 — 4 tools including `submit_plan` |
| `query_profiling_context(aspect)` with 8 aspects | Task 2 — server + Task 3 — tool wrapper |
| `get_kernel_source()` | Task 2 + Task 3 |
| `search_memory(query)` keyword scoring | Task 2 + Task 3 |
| `submit_plan` as terminal tool with Pydantic validation | Task 3 + Task 5 |
| LangGraph sub-graph with ReAct loop | Task 5 — `compile_planner_graph()` |
| Routing: `route_from_agent` + `route_from_tools` | Task 5 — tested explicitly |
| N parallel sub-graph instances via ThreadPoolExecutor | Task 6 — `run_planner_agent` |
| Diversity hint per instance (different NCU rule per plan) | Task 4 — `build_diversity_hint` |
| Cross-iteration direction deduplication | Task 6 — `all_tried_directions` + dedup loop |
| Intra-iteration dedup by direction | Task 6 — `seen` set in `run_planner_agent` |
| `PlannerToolServer` holds immutable data (safe for concurrent reads) | Task 2 — no mutation after `__init__` |
| Remove old hardcoded planner.py | Task 7 |
| Wire into `langgraph_pipeline.py` `generate_plans` node | Task 7 |

**No gaps found.**

**Placeholder scan:** No TBD, TODO, or vague steps. All code is complete.

**Type consistency:**
- `run_planner_agent` signature matches usage in `langgraph_pipeline.py` ✓
- `PlannerToolServer` constructor args match usage in `run_planner_agent` ✓
- `_TOOL_MAP` exported from `tools.py` and used in `graph.py` ✓
- `ALL_TOOLS` exported from `tools.py` and used in `graph.py` `agent_node` ✓
- `planner_subgraph` defined in `graph.py`, imported in `__init__.py`, used in `run_planner_agent` ✓
- `VALID_PLAN` in tests matches `OptimizationPlan` schema fields ✓
