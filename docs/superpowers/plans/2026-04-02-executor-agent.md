# Executor Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the POC `agents/executor.py` (single function, hardcoded prompt) with a LangGraph ReAct sub-graph that mirrors the planner architecture — thin system prompt, tool-driven context access, K-attempt retry inside the graph.

**Architecture:** `ExecutorToolServer` holds immutable kernel/plan/memory data and serves it via `@tool` wrappers. A LangGraph sub-graph (`agent → tools → agent → …`) runs a ReAct loop; `submit_kernel` is intercepted by `tool_node` (not the LLM directly), which validates the submitted kernel, tracks attempts, and writes to state. The graph exits when `kernel_result` is set (success or K exhaustion).

**Tech Stack:** Python, LangGraph, LangChain Core, pytest. Mirrors `kernelopt/kernelopt/agents/planner/` exactly.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `kernelopt/kernelopt/agents/executor/__init__.py` | Exports `run_executor_agent()` |
| Create | `kernelopt/kernelopt/agents/executor/tools.py` | `ExecutorToolServer` + `@tool` wrappers + `submit_kernel` |
| Create | `kernelopt/kernelopt/agents/executor/prompts.py` | `SYSTEM_PROMPT` + `build_initial_message()` |
| Create | `kernelopt/kernelopt/agents/executor/graph.py` | `ExecutorState`, nodes, routing, `compile_executor_graph()` |
| Create | `kernelopt/tests/agents/test_executor_tools.py` | Tests for ToolServer + tool wrappers |
| Create | `kernelopt/tests/agents/test_executor_graph.py` | Tests for routing, tool_node, integration |
| Delete | `kernelopt/kernelopt/agents/executor.py` | Replaced by the package |
| Modify | `kernelopt/kernelopt/langgraph_pipeline.py` | Use `run_executor_agent` instead of `run_executor` |

---

## Task 1: tools.py — ExecutorToolServer + @tool wrappers

**Files:**
- Create: `kernelopt/kernelopt/agents/executor/tools.py`
- Create: `kernelopt/tests/agents/test_executor_tools.py`

- [ ] **Step 1: Write failing tests**

Create `kernelopt/tests/agents/test_executor_tools.py`:

```python
# kernelopt/tests/agents/test_executor_tools.py
import pytest
from langchain_core.runnables import RunnableConfig
from kernelopt.agents.executor.tools import (
    ExecutorToolServer,
    get_kernel_source as tool_get_source,
    get_optimization_plan as tool_get_plan,
    search_memory as tool_search,
    submit_kernel as tool_submit,
    _TOOL_MAP,
    ALL_TOOLS,
)

TRITON_KERNEL = """\
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x = tl.load(x_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE))
    tl.store(y_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE), x)
"""

SAMPLE_PLAN = {
    "plan_id": "e001",
    "direction": "reduce_register_pressure",
    "framework": "triton",
    "step": {"action": "reduce", "target": "BLOCK_SIZE", "change": "Reduce BLOCK_SIZE from 1024 to 512"},
    "correctness_constraints": ["output must match reference"],
    "confidence": "high",
    "confidence_reason": "NCU rule",
}


def _config(server):
    return RunnableConfig(configurable={"tool_server": server})


# --- ExecutorToolServer ---

def test_get_kernel_source():
    server = ExecutorToolServer(TRITON_KERNEL, SAMPLE_PLAN, [])
    assert server.get_kernel_source() == TRITON_KERNEL


def test_get_optimization_plan():
    server = ExecutorToolServer(TRITON_KERNEL, SAMPLE_PLAN, [])
    result = server.get_optimization_plan()
    assert result["plan_id"] == "e001"
    assert result["direction"] == "reduce_register_pressure"


def test_search_memory_ranks_by_keyword(sample_memory):
    server = ExecutorToolServer(TRITON_KERNEL, SAMPLE_PLAN, sample_memory)
    result = server.search_memory("register pressure occupancy")
    assert isinstance(result, list)
    assert len(result) <= 3
    assert result[0]["direction"] == "reduce_tile_size_for_occupancy"


def test_search_memory_empty():
    server = ExecutorToolServer(TRITON_KERNEL, SAMPLE_PLAN, [])
    assert server.search_memory("anything") == []


def test_search_memory_returns_at_most_three():
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
    server = ExecutorToolServer(TRITON_KERNEL, SAMPLE_PLAN, memory)
    assert len(server.search_memory("foo bar")) <= 3


# --- @tool wrappers ---

def test_tool_get_kernel_source():
    server = ExecutorToolServer(TRITON_KERNEL, SAMPLE_PLAN, [])
    result = tool_get_source.invoke({}, config=_config(server))
    assert "@triton.jit" in result


def test_tool_get_optimization_plan():
    server = ExecutorToolServer(TRITON_KERNEL, SAMPLE_PLAN, [])
    result = tool_get_plan.invoke({}, config=_config(server))
    assert result["direction"] == "reduce_register_pressure"


def test_tool_search_memory(sample_memory):
    server = ExecutorToolServer(TRITON_KERNEL, SAMPLE_PLAN, sample_memory)
    result = tool_search.invoke({"query": "register pressure"}, config=_config(server))
    assert isinstance(result, list)


def test_tool_map_excludes_submit_kernel():
    assert "get_kernel_source" in _TOOL_MAP
    assert "get_optimization_plan" in _TOOL_MAP
    assert "search_memory" in _TOOL_MAP
    assert "submit_kernel" not in _TOOL_MAP  # intercepted by tool_node


def test_submit_kernel_tool_has_correct_name():
    assert tool_submit.name == "submit_kernel"


def test_all_tools_has_four_entries():
    assert len(ALL_TOOLS) == 4
    names = {t.name for t in ALL_TOOLS}
    assert names == {"get_kernel_source", "get_optimization_plan", "search_memory", "submit_kernel"}
```

- [ ] **Step 2: Run tests — expect ImportError (module does not exist yet)**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && python -m pytest kernelopt/tests/agents/test_executor_tools.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'kernelopt.agents.executor'`

- [ ] **Step 3: Create the package directory and tools.py**

Create `kernelopt/kernelopt/agents/executor/__init__.py` (empty for now):

```python
# kernelopt/kernelopt/agents/executor/__init__.py
```

Create `kernelopt/kernelopt/agents/executor/tools.py`:

```python
# kernelopt/kernelopt/agents/executor/tools.py
"""ExecutorToolServer and LangChain tool definitions for the Executor Agent."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool


class ExecutorToolServer:
    """Holds immutable kernel/plan/memory data and serves tool calls for the executor sub-graph.

    Thread-safe for concurrent reads — no mutation after construction.
    """

    def __init__(
        self,
        kernel_source: str,
        optimization_plan: dict,
        memory: list[dict],
    ) -> None:
        self.kernel_source = kernel_source
        self.optimization_plan = optimization_plan
        self.memory = memory

    def get_kernel_source(self) -> str:
        """Return the full kernel source to be optimized."""
        return self.kernel_source

    def get_optimization_plan(self) -> dict:
        """Return the optimization plan the executor must implement."""
        return self.optimization_plan

    def search_memory(self, query: str) -> list[dict]:
        """Return up to 3 memory items most relevant to the query.

        Keyword matching is substring-based. Empty query returns first 3 items.
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


# ---------------------------------------------------------------------------
# LangChain @tool wrappers — delegate to ExecutorToolServer via configurable
# ---------------------------------------------------------------------------

@tool
def get_kernel_source(
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Return the full kernel source code that needs to be optimized.

    Call this to read the current kernel before making changes.
    """
    server: ExecutorToolServer = config["configurable"]["tool_server"]
    return server.get_kernel_source()


@tool
def get_optimization_plan(
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> dict:
    """Return the optimization plan to implement.

    Contains: plan_id, direction, framework, step.change, step.target,
    step.predicted_effect, correctness_constraints, evidence.
    """
    server: ExecutorToolServer = config["configurable"]["tool_server"]
    return server.get_optimization_plan()


@tool
def search_memory(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> list[dict]:
    """Search past optimization experience for relevant patterns.

    query — natural language, e.g. "register pressure triton" or "tile size failure".
    Returns up to 3 memory items ranked by keyword relevance.
    Each item includes: direction, strategy_description, applicable_when,
    do_not_apply_when, rewrite_type (positive/negative/implementation_flaw), speedup.
    """
    server: ExecutorToolServer = config["configurable"]["tool_server"]
    return server.search_memory(query)


@tool
def submit_kernel(kernel_source: str, change_summary: str) -> str:
    """Submit the complete optimized kernel source.

    kernel_source  — the full file with ONLY the plan's change applied.
    change_summary — one sentence describing exactly what changed.

    Call this when you have implemented the optimization. If validation fails,
    diagnose the error in the response and resubmit a corrected version.
    """
    # Body never called — tool_node intercepts submit_kernel and handles
    # validation + state update directly (same pattern as submit_plan in planner).
    return "received"


# Map of tool name → tool function for tool_node dispatch.
# submit_kernel is intentionally excluded — handled separately in tool_node.
_TOOL_MAP: dict[str, Any] = {
    "get_kernel_source": get_kernel_source,
    "get_optimization_plan": get_optimization_plan,
    "search_memory": search_memory,
}

# All tools to bind to the LLM (including submit_kernel).
ALL_TOOLS = [get_kernel_source, get_optimization_plan, search_memory, submit_kernel]
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && python -m pytest kernelopt/tests/agents/test_executor_tools.py -v
```

Expected: all 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && git add kernelopt/kernelopt/agents/executor/__init__.py kernelopt/kernelopt/agents/executor/tools.py kernelopt/tests/agents/test_executor_tools.py && git commit -m "feat: add ExecutorToolServer and @tool wrappers for executor agent"
```

---

## Task 2: prompts.py — system prompt + initial message builder

**Files:**
- Create: `kernelopt/kernelopt/agents/executor/prompts.py`
- Tests inline in: `kernelopt/tests/agents/test_executor_tools.py` (append to existing file)

- [ ] **Step 1: Append prompt tests to test_executor_tools.py**

Append to `kernelopt/tests/agents/test_executor_tools.py`:

```python
# --- prompts ---

from langchain_core.messages import HumanMessage
from kernelopt.agents.executor.prompts import SYSTEM_PROMPT, build_initial_message


def test_system_prompt_contains_role():
    assert "Executor Agent" in SYSTEM_PROMPT


def test_system_prompt_mentions_submit_kernel():
    assert "submit_kernel" in SYSTEM_PROMPT


def test_system_prompt_has_no_hardcoded_framework_rules():
    # Must NOT contain framework-specific transformation recipes
    assert "__launch_bounds__" not in SYSTEM_PROMPT
    assert "tl.constexpr" not in SYSTEM_PROMPT
    assert "BLOCK_M" not in SYSTEM_PROMPT


def test_build_initial_message_is_human_message():
    msg = build_initial_message("reduce_register_pressure", K=3)
    assert isinstance(msg, HumanMessage)


def test_build_initial_message_contains_direction():
    msg = build_initial_message("reduce_register_pressure", K=3)
    assert "reduce_register_pressure" in msg.content


def test_build_initial_message_contains_attempt_count():
    msg = build_initial_message("increase_tile_k", K=2)
    assert "2" in msg.content
```

- [ ] **Step 2: Run tests — expect ImportError on prompts**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && python -m pytest kernelopt/tests/agents/test_executor_tools.py -v -k "prompt or system or initial" 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'SYSTEM_PROMPT'`

- [ ] **Step 3: Create prompts.py**

Create `kernelopt/kernelopt/agents/executor/prompts.py`:

```python
# kernelopt/kernelopt/agents/executor/prompts.py
"""System prompt and initial message builder for the Executor Agent."""

from __future__ import annotations

from langchain_core.messages import HumanMessage


SYSTEM_PROMPT = """\
You are the Executor Agent in a GPU kernel optimization pipeline.

You receive an optimization plan and a kernel source file.
Your job is to implement exactly the change described in the plan.

Use your tools to understand the context:
  get_optimization_plan()  — read the plan you must implement
  get_kernel_source()      — read the current kernel source
  search_memory(query)     — check past experience for relevant patterns

When you have implemented the change, call submit_kernel(kernel_source, change_summary).
  kernel_source  — the complete modified file with ONLY the plan's change applied
  change_summary — one sentence describing what changed

If submit_kernel returns a validation error, diagnose what went wrong and resubmit.
"""


def build_initial_message(plan_direction: str, K: int) -> HumanMessage:
    """Build the opening human message for the executor sub-graph."""
    return HumanMessage(content=(
        f"Implement the optimization plan.\n\n"
        f"Direction: {plan_direction}\n"
        f"You have up to {K} attempts via submit_kernel."
    ))
```

- [ ] **Step 4: Run all tests — expect all pass**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && python -m pytest kernelopt/tests/agents/test_executor_tools.py -v
```

Expected: all 19 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && git add kernelopt/kernelopt/agents/executor/prompts.py kernelopt/tests/agents/test_executor_tools.py && git commit -m "feat: add executor agent system prompt and initial message builder"
```

---

## Task 3: graph.py — state, routing, tool_node, compile_executor_graph

**Files:**
- Create: `kernelopt/kernelopt/agents/executor/graph.py`
- Create: `kernelopt/tests/agents/test_executor_graph.py`

- [ ] **Step 1: Write failing graph tests**

Create `kernelopt/tests/agents/test_executor_graph.py`:

```python
# kernelopt/tests/agents/test_executor_graph.py
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from kernelopt.agents.executor.graph import route_from_agent, route_from_tools, tool_node
from kernelopt.agents.executor.tools import ExecutorToolServer

TRITON_KERNEL = """\
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x = tl.load(x_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE))
    tl.store(y_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE), x)
"""

CUDA_KERNEL = """\
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""

SAMPLE_PLAN = {
    "plan_id": "e001",
    "direction": "reduce_register_pressure",
    "framework": "triton",
    "step": {"action": "reduce", "target": "BLOCK_SIZE", "change": "Reduce BLOCK_SIZE from 1024 to 512"},
    "correctness_constraints": [],
    "confidence": "high",
    "confidence_reason": "NCU rule",
}


def _config(kernel=TRITON_KERNEL, plan=SAMPLE_PLAN, memory=None, K=2):
    server = ExecutorToolServer(kernel, plan, memory or [])
    return RunnableConfig(configurable={
        "tool_server": server,
        "K": K,
        "plan_id": plan["plan_id"],
        "original_kernel": kernel,
        "llm": None,
        "llm_config": {},
    })


# --- route_from_agent ---

def test_route_from_agent_no_tool_calls_goes_to_end():
    msg = AIMessage(content="I am done.")
    state = {"messages": [HumanMessage(content="go"), msg], "kernel_result": None, "attempt": 0}
    assert route_from_agent(state) == END


def test_route_from_agent_with_tool_calls_goes_to_tools():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "get_kernel_source", "args": {}, "id": "tc1"}],
    )
    state = {"messages": [HumanMessage(content="go"), msg], "kernel_result": None, "attempt": 0}
    assert route_from_agent(state) == "tools"


def test_route_from_agent_empty_tool_calls_list_goes_to_end():
    msg = AIMessage(content="Done.", tool_calls=[])
    state = {"messages": [HumanMessage(content="go"), msg], "kernel_result": None, "attempt": 0}
    assert route_from_agent(state) == END


# --- route_from_tools ---

def test_route_from_tools_kernel_result_set_goes_to_end():
    state = {
        "messages": [],
        "kernel_result": {"kernel_source": TRITON_KERNEL, "execution_result": {"status": "success"}},
        "attempt": 1,
    }
    assert route_from_tools(state) == END


def test_route_from_tools_no_kernel_result_goes_to_agent():
    state = {"messages": [], "kernel_result": None, "attempt": 0}
    assert route_from_tools(state) == "agent"


# --- tool_node: regular tool calls ---

def test_tool_node_get_kernel_source_returns_tool_message():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "get_kernel_source", "args": {}, "id": "tc1"}],
    )
    state = {"messages": [HumanMessage(content="go"), msg], "kernel_result": None, "attempt": 0}
    result = tool_node(state, _config())
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ToolMessage)
    assert "@triton.jit" in result["messages"][0].content
    assert result.get("kernel_result") is None  # not set by a non-submit call


def test_tool_node_get_optimization_plan_returns_tool_message():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "get_optimization_plan", "args": {}, "id": "tc2"}],
    )
    state = {"messages": [HumanMessage(content="go"), msg], "kernel_result": None, "attempt": 0}
    result = tool_node(state, _config())
    assert isinstance(result["messages"][0], ToolMessage)
    assert "reduce_register_pressure" in result["messages"][0].content


def test_tool_node_unknown_tool_returns_error_message():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "nonexistent_tool", "args": {}, "id": "tc3"}],
    )
    state = {"messages": [HumanMessage(content="go"), msg], "kernel_result": None, "attempt": 0}
    result = tool_node(state, _config())
    assert "Unknown tool" in result["messages"][0].content


# --- tool_node: submit_kernel intercept ---

def test_tool_node_submit_valid_triton_kernel_succeeds():
    msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "submit_kernel",
            "args": {"kernel_source": TRITON_KERNEL, "change_summary": "Reduced BLOCK_SIZE to 512"},
            "id": "tc4",
        }],
    )
    state = {"messages": [HumanMessage(content="go"), msg], "kernel_result": None, "attempt": 0}
    result = tool_node(state, _config(K=2))

    assert result["attempt"] == 1
    assert result["kernel_result"] is not None
    er = result["kernel_result"]["execution_result"]
    assert er["status"] == "success"
    assert er["framework"] == "triton"
    assert er["ready_for_profiling"] is True
    assert er["correctness"] == "unverified"
    assert er["plan_id"] == "e001"
    assert isinstance(result["messages"][0], ToolMessage)


def test_tool_node_submit_valid_cuda_kernel_infers_cuda():
    msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "submit_kernel",
            "args": {"kernel_source": CUDA_KERNEL, "change_summary": "tiled access"},
            "id": "tc5",
        }],
    )
    state = {"messages": [HumanMessage(content="go"), msg], "kernel_result": None, "attempt": 0}
    result = tool_node(state, _config(kernel=CUDA_KERNEL, K=1))
    assert result["kernel_result"]["execution_result"]["framework"] == "cuda"


def test_tool_node_submit_empty_kernel_under_K_continues():
    msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "submit_kernel",
            "args": {"kernel_source": "", "change_summary": "oops"},
            "id": "tc6",
        }],
    )
    state = {"messages": [HumanMessage(content="go"), msg], "kernel_result": None, "attempt": 0}
    result = tool_node(state, _config(K=2))

    # attempt incremented but kernel_result NOT set — loop continues
    assert result["attempt"] == 1
    assert result.get("kernel_result") is None
    assert isinstance(result["messages"][0], ToolMessage)
    content = result["messages"][0].content.lower()
    assert "validation failed" in content or "empty" in content


def test_tool_node_submit_empty_kernel_at_K_exhausts():
    msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "submit_kernel",
            "args": {"kernel_source": "", "change_summary": "still broken"},
            "id": "tc7",
        }],
    )
    # attempt is already 1, K=2 → this is attempt 2 → exhausted
    state = {"messages": [HumanMessage(content="go"), msg], "kernel_result": None, "attempt": 1}
    result = tool_node(state, _config(K=2))

    assert result["attempt"] == 2
    assert result["kernel_result"] is not None
    er = result["kernel_result"]["execution_result"]
    assert er["status"] == "implementation_failure"
    assert er["ready_for_profiling"] is False
    # Returns original kernel unchanged
    assert result["kernel_result"]["kernel_source"] == TRITON_KERNEL


def test_tool_node_submit_no_marker_under_K_continues():
    no_marker = "def not_a_kernel(): return 42\n"
    msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "submit_kernel",
            "args": {"kernel_source": no_marker, "change_summary": "wrong"},
            "id": "tc8",
        }],
    )
    state = {"messages": [HumanMessage(content="go"), msg], "kernel_result": None, "attempt": 0}
    result = tool_node(state, _config(K=3))
    assert result.get("kernel_result") is None
    assert result["attempt"] == 1
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && python -m pytest kernelopt/tests/agents/test_executor_graph.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'route_from_agent' from 'kernelopt.agents.executor.graph'`

- [ ] **Step 3: Create graph.py**

Create `kernelopt/kernelopt/agents/executor/graph.py`:

```python
# kernelopt/kernelopt/agents/executor/graph.py
"""Executor Agent LangGraph sub-graph definition."""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from kernelopt.agents.executor.prompts import SYSTEM_PROMPT
from kernelopt.agents.executor.tools import ALL_TOOLS, _TOOL_MAP


class ExecutorState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    kernel_result: dict | None   # set on success or K exhaustion; None until then
    attempt: int                 # number of submit_kernel calls made so far


_KERNEL_MARKERS = ["@triton.jit", "__global__", "@helion.kernel", "@hl.kernel"]


def _infer_framework(kernel_source: str) -> str:
    if "@triton.jit" in kernel_source:
        return "triton"
    if "__global__" in kernel_source:
        return "cuda"
    if "@helion.kernel" in kernel_source or "@hl.kernel" in kernel_source:
        return "helion"
    return "unknown"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def agent_node(state: ExecutorState, config: RunnableConfig) -> dict:
    """Call the LLM with tools bound. Accepts a pre-built LLM via configurable for testing."""
    from kernelopt.llm import build_llm

    llm = config["configurable"].get("llm")
    if llm is None:
        llm_cfg = config["configurable"]["llm_config"]
        llm = build_llm(**llm_cfg)

    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: ExecutorState, config: RunnableConfig) -> dict:
    """Execute tool calls from the last agent message.

    submit_kernel is intercepted: validates the kernel source, increments the attempt
    counter, and writes kernel_result to state on success or K-exhaustion.
    All other tools are executed normally via _TOOL_MAP.
    """
    last = state["messages"][-1]
    K: int = config["configurable"]["K"]
    plan_id: str = config["configurable"]["plan_id"]
    original_kernel: str = config["configurable"]["original_kernel"]

    tool_messages: list[ToolMessage] = []
    new_kernel_result: dict | None = None
    new_attempt: int = state.get("attempt", 0)

    for tc in last.tool_calls:
        name = tc["name"]
        args = tc["args"]
        tc_id = tc["id"]

        if name == "submit_kernel":
            new_attempt = state.get("attempt", 0) + 1
            kernel_source: str = args.get("kernel_source", "")
            change_summary: str = args.get("change_summary", "")

            has_marker = any(m in kernel_source for m in _KERNEL_MARKERS)
            is_valid = bool(kernel_source.strip()) and has_marker

            if is_valid:
                framework = _infer_framework(kernel_source)
                orig_lines = set(original_kernel.splitlines())
                new_lines = set(kernel_source.splitlines())
                lines_changed = len(new_lines - orig_lines)

                new_kernel_result = {
                    "kernel_source": kernel_source,
                    "execution_result": {
                        "plan_id": plan_id,
                        "attempt": new_attempt,
                        "framework": framework,
                        "status": "success",
                        "correctness": "unverified",
                        "change_summary": change_summary,
                        "lines_changed": lines_changed,
                        "ready_for_profiling": True,
                    },
                }
                tool_messages.append(
                    ToolMessage(content="Kernel submitted successfully.", tool_call_id=tc_id)
                )

            elif new_attempt >= K:
                reason = (
                    "Submitted kernel is empty."
                    if not kernel_source.strip()
                    else f"No kernel marker found. Expected one of: {_KERNEL_MARKERS}"
                )
                new_kernel_result = {
                    "kernel_source": original_kernel,
                    "execution_result": {
                        "plan_id": plan_id,
                        "attempt": new_attempt,
                        "framework": "unknown",
                        "status": "implementation_failure",
                        "reason": reason,
                        "ready_for_profiling": False,
                    },
                }
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Validation failed (attempt {new_attempt}/{K}): {reason}. "
                            f"All attempts exhausted."
                        ),
                        tool_call_id=tc_id,
                    )
                )

            else:
                reason = (
                    "Submitted kernel is empty."
                    if not kernel_source.strip()
                    else f"No kernel marker found. Expected one of: {_KERNEL_MARKERS}"
                )
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Validation failed (attempt {new_attempt}/{K}): {reason}. "
                            f"Diagnose and resubmit a corrected version."
                        ),
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

    updates: dict = {"messages": tool_messages, "attempt": new_attempt}
    if new_kernel_result is not None:
        updates["kernel_result"] = new_kernel_result
    return updates


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_from_agent(state: ExecutorState) -> str:
    """Route to tools if the agent made tool calls, otherwise END (fallback)."""
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END
    return "tools"


def route_from_tools(state: ExecutorState) -> str:
    """Route to END if kernel_result is set, otherwise back to agent."""
    if state.get("kernel_result") is not None:
        return END
    return "agent"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def compile_executor_graph():
    """Return a compiled LangGraph sub-graph for one executor instance."""
    graph = StateGraph(ExecutorState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_from_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_from_tools, {"agent": "agent", END: END})

    return graph.compile()


executor_subgraph = compile_executor_graph()
```

- [ ] **Step 4: Run graph tests — expect all pass**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && python -m pytest kernelopt/tests/agents/test_executor_graph.py -v
```

Expected: all 15 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && git add kernelopt/kernelopt/agents/executor/graph.py kernelopt/tests/agents/test_executor_graph.py && git commit -m "feat: add executor agent LangGraph sub-graph with ReAct loop and K-attempt retry"
```

---

## Task 4: __init__.py — run_executor_agent + integration tests

**Files:**
- Modify: `kernelopt/kernelopt/agents/executor/__init__.py`
- Append to: `kernelopt/tests/agents/test_executor_graph.py`

- [ ] **Step 1: Append integration tests to test_executor_graph.py**

Append to `kernelopt/tests/agents/test_executor_graph.py`:

```python
# --- run_executor_agent integration ---

from unittest.mock import MagicMock
from kernelopt.agents.executor import run_executor_agent


def _make_mock_llm(tool_calls_sequence):
    """Return a mock LLM that emits AIMessages with the given tool_calls in sequence."""
    mock_llm = MagicMock()
    responses = [
        AIMessage(content="", tool_calls=tcs) for tcs in tool_calls_sequence
    ]
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = responses
    return mock_llm


def test_run_executor_agent_success_on_first_attempt(sample_memory):
    mock_llm = _make_mock_llm([
        [{"name": "submit_kernel", "args": {"kernel_source": TRITON_KERNEL, "change_summary": "Reduced BLOCK_SIZE"}, "id": "tc1"}],
    ])
    kernel, result = run_executor_agent(
        kernel_source=TRITON_KERNEL,
        optimization_plan=SAMPLE_PLAN,
        memory=sample_memory,
        K=2,
        llm_config={"model": "unused", "api_key": None, "project_id": None, "region": None},
        llm_override=mock_llm,
    )
    assert "@triton.jit" in kernel
    assert result["status"] == "success"
    assert result["ready_for_profiling"] is True
    assert result["framework"] == "triton"
    assert result["attempt"] == 1


def test_run_executor_agent_retries_then_succeeds(sample_memory):
    """Agent fails once (empty kernel), then succeeds on second attempt."""
    mock_llm = _make_mock_llm([
        # First submit: empty kernel (fails validation)
        [{"name": "submit_kernel", "args": {"kernel_source": "", "change_summary": "bad"}, "id": "tc1"}],
        # Agent tries again: valid kernel
        [{"name": "submit_kernel", "args": {"kernel_source": TRITON_KERNEL, "change_summary": "fixed"}, "id": "tc2"}],
    ])
    kernel, result = run_executor_agent(
        kernel_source=TRITON_KERNEL,
        optimization_plan=SAMPLE_PLAN,
        memory=sample_memory,
        K=2,
        llm_config={"model": "unused", "api_key": None, "project_id": None, "region": None},
        llm_override=mock_llm,
    )
    assert result["status"] == "success"
    assert result["attempt"] == 2


def test_run_executor_agent_exhausts_K_returns_failure(sample_memory):
    """Agent submits empty kernel K times → implementation_failure returned."""
    mock_llm = _make_mock_llm([
        [{"name": "submit_kernel", "args": {"kernel_source": "", "change_summary": "bad"}, "id": "tc1"}],
        [{"name": "submit_kernel", "args": {"kernel_source": "", "change_summary": "bad"}, "id": "tc2"}],
    ])
    kernel, result = run_executor_agent(
        kernel_source=TRITON_KERNEL,
        optimization_plan=SAMPLE_PLAN,
        memory=sample_memory,
        K=2,
        llm_config={"model": "unused", "api_key": None, "project_id": None, "region": None},
        llm_override=mock_llm,
    )
    assert result["status"] == "implementation_failure"
    assert result["ready_for_profiling"] is False
    assert kernel == TRITON_KERNEL  # original returned unchanged


def test_run_executor_agent_no_submit_returns_failure(sample_memory):
    """Agent emits no tool calls → graph ends without kernel_result → failure."""
    mock_llm = _make_mock_llm([[]])  # empty tool_calls list → route_from_agent → END
    kernel, result = run_executor_agent(
        kernel_source=TRITON_KERNEL,
        optimization_plan=SAMPLE_PLAN,
        memory=sample_memory,
        K=2,
        llm_config={"model": "unused", "api_key": None, "project_id": None, "region": None},
        llm_override=mock_llm,
    )
    assert result["status"] == "implementation_failure"
    assert result["ready_for_profiling"] is False
```

- [ ] **Step 2: Run integration tests — expect ImportError**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && python -m pytest kernelopt/tests/agents/test_executor_graph.py -v -k "run_executor_agent" 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'run_executor_agent'`

- [ ] **Step 3: Implement __init__.py**

Replace `kernelopt/kernelopt/agents/executor/__init__.py` with:

```python
# kernelopt/kernelopt/agents/executor/__init__.py
"""Executor Agent package.

Exports:
    run_executor_agent — runs one executor sub-graph instance
    executor_subgraph  — the compiled single-instance sub-graph (for testing/reuse)
"""

from __future__ import annotations

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
) -> tuple[str, dict]:
    """Run the executor sub-graph to apply one optimization plan.

    Args:
        kernel_source:      Current kernel source to be optimized.
        optimization_plan:  Plan dict from the planner agent.
        memory:             Optimization memory items from prior iterations.
        K:                  Maximum submit_kernel attempts allowed.
        llm_config:         Dict with keys model, api_key, project_id, region.
        llm_override:       Optional pre-built LLM (used in tests to avoid API calls).

    Returns:
        (optimized_kernel_source, execution_result) — drop-in for old run_executor().
    """
    plan_direction = optimization_plan.get("direction", "unknown")
    plan_id = optimization_plan.get("plan_id", "")

    server = ExecutorToolServer(kernel_source, optimization_plan, memory)
    initial_msg = build_initial_message(plan_direction, K)

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
        import traceback
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
```

- [ ] **Step 4: Run all executor tests — expect all pass**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && python -m pytest kernelopt/tests/agents/test_executor_tools.py kernelopt/tests/agents/test_executor_graph.py -v
```

Expected: all tests PASS (19 + 15 + 4 integration = 38 total).

- [ ] **Step 5: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && git add kernelopt/kernelopt/agents/executor/__init__.py kernelopt/tests/agents/test_executor_graph.py && git commit -m "feat: add run_executor_agent public interface with K-attempt retry sub-graph"
```

---

## Task 5: Update langgraph_pipeline.py

**Files:**
- Modify: `kernelopt/kernelopt/langgraph_pipeline.py` (lines 19, 160–284)

- [ ] **Step 1: Update the import at the top of langgraph_pipeline.py**

In `kernelopt/kernelopt/langgraph_pipeline.py`, replace line 19:

```python
from kernelopt.agents.executor import run_executor
```

with:

```python
from kernelopt.agents.executor import run_executor_agent
```

- [ ] **Step 2: Replace execute_plans node body**

Replace the entire `execute_plans` function in `langgraph_pipeline.py` (find it by the `def execute_plans(state: GraphState)` signature) with:

```python
def execute_plans(state: GraphState) -> dict:
    """Apply each plan to produce optimized kernels and check correctness."""
    cfg = state["config"]
    plans = state["plans"]
    current_kernel = state["best_kernel"]
    framework = state["framework"]
    output_dir = Path(state["output_dir"])
    ext = state["ext"]
    iteration = state.get("iteration", 0)
    kernel_basename = state["kernel_basename"]
    kernel_path = state["kernel_path"]

    llm_config = {
        "model": cfg.executor_model or cfg.model,
        "api_key": cfg.api_key,
        "project_id": cfg.project_id,
        "region": cfg.region,
    }

    candidates: list[PlanCandidate] = []
    correctness_failure_items: list[dict] = []

    for plan_letter, plan in plans:
        direction = plan.get("direction", "unknown")
        safe_dir = direction.replace(" ", "_")[:40]
        file_stem = f"{kernel_basename}_iter{iteration + 1}_plan{plan_letter}_{safe_dir}"

        console.print(f"  [dim]Executing plan {plan_letter} ({direction})...[/]")
        try:
            optimized_source, execution_result = run_executor_agent(
                kernel_source=current_kernel,
                optimization_plan=plan,
                memory=state.get("optimization_memory", []),
                K=cfg.K,
                llm_config=llm_config,
            )
        except Exception as e:
            import traceback
            console.print(f"    [red]Executor failed (plan {plan_letter}): {e}[/]")
            continue

        console.print(f"  [dim]Execution status: {execution_result.get('status')}[/]")
        console.print(f"  [dim]Ready for profiling: {execution_result.get('ready_for_profiling')}[/]")
        console.print(f"  [dim]Lines changed: {execution_result.get('lines_changed')}[/]")

        if not execution_result.get("ready_for_profiling", False):
            reason = execution_result.get("reason", execution_result.get("status", "failed"))
            console.print(f"    [red]Plan {plan_letter}: {reason}[/]")
            continue

        # Save kernel file
        kernel_file = str(output_dir / f"{file_stem}{ext}")
        Path(kernel_file).write_text(optimized_source)

        # Correctness check
        console.print(f"  [dim]Running correctness check...[/]")
        passed, msg = check_correctness(
            kernel_path, optimized_source, framework,
            use_agentic=cfg.agentic_correctness,
            api_key=cfg.api_key,
            project_id=cfg.project_id,
            region=cfg.region,
            kernel_args=cfg.kernel_args,
        )
        execution_result["correctness"] = "passed" if passed else "failed"

        status = "[green]passed[/]" if passed else "[red]failed[/]"
        console.print(f"    Plan {plan_letter}: correctness {status}")

        if not passed:
            console.print(f"    [dim]Failure: {msg[:300]}[/]")
            import uuid as _uuid
            correctness_failure_items.append({
                "item_id": _uuid.uuid4().hex[:8],
                "iteration": iteration + 1,
                "speedup": 0.0,
                "rewrite_type": "implementation_flaw",
                "framework": framework,
                "direction": direction,
                "profiling_signal": {
                    "ncu_rule_triggered": plan.get("evidence", {}).get("ncu_rule"),
                    "key_metric_before": {},
                    "key_metric_after": {},
                },
                "strategy_title": f"{direction} — correctness failure",
                "strategy_description": (
                    f"Plan '{direction}' was applied but failed correctness. "
                    f"Error: {msg[:300]}"
                ),
                "slow_pseudocode": "",
                "fast_pseudocode": "",
                "applicable_when": plan.get("evidence", {}).get("reasoning", ""),
                "do_not_apply_when": f"Previous implementation failed: {msg[:200]}",
                "framework_notes": None,
            })
            continue

        candidate = PlanCandidate(
            plan_id=plan.get("plan_id", ""),
            plan_letter=plan_letter,
            direction=direction,
            optimization_plan=plan,
            optimized_kernel=optimized_source,
            execution_result=execution_result,
            correctness_passed=passed,
            correctness_message=msg,
        )
        candidates.append(candidate)

    return {"candidates": candidates, "correctness_failure_items": correctness_failure_items}
```

- [ ] **Step 3: Verify existing tests still pass**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && python -m pytest kernelopt/tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Verify the import works in the pipeline module**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && python -c "from kernelopt.langgraph_pipeline import execute_plans; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && git add kernelopt/kernelopt/langgraph_pipeline.py && git commit -m "feat: wire run_executor_agent into pipeline, remove old run_executor call and retry loop"
```

---

## Task 6: Delete old executor.py and run full test suite

**Files:**
- Delete: `kernelopt/kernelopt/agents/executor.py`

- [ ] **Step 1: Delete the old executor module**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && git rm kernelopt/kernelopt/agents/executor.py
```

- [ ] **Step 2: Check nothing still imports run_executor**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && grep -r "run_executor\b" kernelopt/ --include="*.py"
```

Expected: no output (zero matches). If matches are found, update those imports.

- [ ] **Step 3: Run the full test suite**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && python -m pytest kernelopt/tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /workspaces/pytorch-devcontainers/TOOL && git commit -m "chore: delete old executor.py POC, replaced by executor/ package"
```
