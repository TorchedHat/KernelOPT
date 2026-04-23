# kernelopt/tests/agents/test_executor_tools.py
from __future__ import annotations

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
    assert len(server.search_memory("foo bar")) == 3


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
    result = tool_search.invoke({"query": "register pressure occupancy"}, config=_config(server))
    assert isinstance(result, list)
    assert result[0]["direction"] == "reduce_tile_size_for_occupancy"


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


def test_build_initial_message_without_compile_error_has_no_nvcc_section():
    msg = build_initial_message("tiling", K=2)
    assert "nvcc" not in msg.content
    assert "compilation" not in msg.content.lower()


def test_build_initial_message_with_compile_error_includes_error():
    msg = build_initial_message("tiling", K=2, compile_error="kernel.cu(5): error: expected ';'")
    assert "kernel.cu(5): error: expected ';'" in msg.content


def test_build_initial_message_with_compile_error_instructs_fix():
    msg = build_initial_message("tiling", K=2, compile_error="kernel.cu(3): error: foo")
    assert "compile" in msg.content.lower() or "nvcc" in msg.content.lower()
