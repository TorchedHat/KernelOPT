"""
Tests for profiler agent tools.

Tests ProfilerToolServer, read_kernel_source, and submit_ncu_config tools.
"""
import pytest
from langchain_core.runnables import RunnableConfig

from kernelopt.agents.profiler.tools import (
    ALL_TOOLS,
    ProfilerToolServer,
    _TOOL_MAP,
    read_kernel_source,
    submit_ncu_config,
)

TRITON_SOURCE = """\
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    tl.store(y_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE),
             tl.load(x_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE)))

def run_softmax(x, y, n_cols):
    softmax_kernel[(x.shape[0],)](x, y, n_cols, BLOCK_SIZE=1024)
"""


def _config(source=TRITON_SOURCE, path="fake.py"):
    server = ProfilerToolServer(path, source)
    return RunnableConfig(configurable={"tool_server": server, "llm": None, "llm_config": {}})


# --- ProfilerToolServer ---

def test_tool_server_stores_source():
    server = ProfilerToolServer("my_kernel.py", TRITON_SOURCE)
    assert server.kernel_source == TRITON_SOURCE
    assert server.kernel_path == "my_kernel.py"


# --- read_kernel_source ---

def test_read_kernel_source_returns_full_source():
    result = read_kernel_source.invoke({}, config=_config())
    assert "@triton.jit" in result
    assert "softmax_kernel" in result
    assert result == TRITON_SOURCE


# --- submit_ncu_config is in ALL_TOOLS ---

def test_all_tools_contains_both_tools():
    names = {t.name for t in ALL_TOOLS}
    assert "read_kernel_source" in names
    assert "submit_ncu_config" in names


def test_tool_map_contains_only_read_kernel_source():
    assert "read_kernel_source" in _TOOL_MAP
    assert "submit_ncu_config" not in _TOOL_MAP


# --- submit_ncu_config schema (via tool metadata) ---

def test_submit_ncu_config_has_correct_args():
    args_schema = submit_ncu_config.args_schema.model_fields
    assert "kernel_name" in args_schema
    assert "launch_skip" in args_schema
    assert "launch_count" in args_schema
    assert "kernel_args" in args_schema
    assert "replay_mode" in args_schema
