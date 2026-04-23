"""
Tests for profiler LangGraph agent.

Tests routing, tool_node, and profiler_subgraph integration.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from kernelopt.agents.profiler.graph import (
    ProfilerState,
    route_from_agent,
    route_from_tools,
    tool_node,
)
from kernelopt.agents.profiler.tools import ProfilerToolServer

TRITON_SOURCE = """\
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    tl.store(y_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE),
             tl.load(x_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE)))
"""

VALID_CONFIG_ARGS = {
    "kernel_name": "softmax_kernel",
    "launch_skip": 0,
    "launch_count": 1,
    "kernel_args": [],
    "replay_mode": "kernel",
}


def _config(source=TRITON_SOURCE):
    server = ProfilerToolServer("fake.py", source)
    return RunnableConfig(configurable={
        "tool_server": server,
        "llm": None,
        "llm_config": {},
    })


# --- route_from_agent ---

def test_route_from_agent_no_tool_calls_goes_to_end():
    msg = AIMessage(content="Done.")
    state: ProfilerState = {"messages": [HumanMessage(content="go"), msg], "ncu_config": None}
    assert route_from_agent(state) == END


def test_route_from_agent_with_tool_calls_goes_to_tools():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "read_kernel_source", "args": {}, "id": "tc1"}],
    )
    state: ProfilerState = {"messages": [HumanMessage(content="go"), msg], "ncu_config": None}
    assert route_from_agent(state) == "tools"


def test_route_from_agent_empty_tool_calls_goes_to_end():
    msg = AIMessage(content="Done.", tool_calls=[])
    state: ProfilerState = {"messages": [HumanMessage(content="go"), msg], "ncu_config": None}
    assert route_from_agent(state) == END


# --- route_from_tools ---

def test_route_from_tools_ncu_config_set_goes_to_end():
    state: ProfilerState = {
        "messages": [],
        "ncu_config": {"kernel_name": "softmax_kernel", "launch_skip": 0,
                       "launch_count": 1, "kernel_args": [], "replay_mode": "kernel"},
    }
    assert route_from_tools(state) == END


def test_route_from_tools_no_config_goes_to_agent():
    state: ProfilerState = {"messages": [], "ncu_config": None}
    assert route_from_tools(state) == "agent"


# --- tool_node: read_kernel_source ---

def test_tool_node_read_kernel_source_returns_source():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "read_kernel_source", "args": {}, "id": "tc1"}],
    )
    state: ProfilerState = {"messages": [HumanMessage(content="go"), msg], "ncu_config": None}
    result = tool_node(state, _config())
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ToolMessage)
    assert "@triton.jit" in result["messages"][0].content
    assert result.get("ncu_config") is None


# --- tool_node: submit_ncu_config ---

def test_tool_node_submit_valid_config_sets_ncu_config():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "submit_ncu_config", "args": VALID_CONFIG_ARGS, "id": "tc2"}],
    )
    state: ProfilerState = {"messages": [HumanMessage(content="go"), msg], "ncu_config": None}
    result = tool_node(state, _config())
    assert result["ncu_config"] is not None
    assert result["ncu_config"]["kernel_name"] == "softmax_kernel"
    assert result["ncu_config"]["launch_count"] == 1
    assert result["ncu_config"]["kernel_args"] == []
    assert result["ncu_config"]["replay_mode"] == "kernel"
    assert "successfully" in result["messages"][0].content


def test_tool_node_submit_empty_kernel_name_fails_validation():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "submit_ncu_config", "args": {
            **VALID_CONFIG_ARGS, "kernel_name": "  "
        }, "id": "tc3"}],
    )
    state: ProfilerState = {"messages": [HumanMessage(content="go"), msg], "ncu_config": None}
    result = tool_node(state, _config())
    assert result.get("ncu_config") is None
    assert "Validation failed" in result["messages"][0].content


def test_tool_node_submit_bad_launch_count_fails_validation():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "submit_ncu_config", "args": {
            **VALID_CONFIG_ARGS, "launch_count": 0
        }, "id": "tc4"}],
    )
    state: ProfilerState = {"messages": [HumanMessage(content="go"), msg], "ncu_config": None}
    result = tool_node(state, _config())
    assert result.get("ncu_config") is None


def test_tool_node_submit_bad_replay_mode_fails_validation():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "submit_ncu_config", "args": {
            **VALID_CONFIG_ARGS, "replay_mode": "fast"
        }, "id": "tc5"}],
    )
    state: ProfilerState = {"messages": [HumanMessage(content="go"), msg], "ncu_config": None}
    result = tool_node(state, _config())
    assert result.get("ncu_config") is None


def test_tool_node_unknown_tool_returns_error():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "nonexistent", "args": {}, "id": "tc6"}],
    )
    state: ProfilerState = {"messages": [HumanMessage(content="go"), msg], "ncu_config": None}
    result = tool_node(state, _config())
    assert "Unknown tool" in result["messages"][0].content


# --- profiler_subgraph integration (mock LLM — no real NCU) ---

def _make_mock_llm(tool_calls_sequence):
    mock = MagicMock()
    responses = [AIMessage(content="", tool_calls=tcs) for tcs in tool_calls_sequence]
    mock.bind_tools.return_value = mock
    mock.invoke.side_effect = responses
    return mock


def test_profiler_subgraph_reads_source_then_submits(tmp_path):
    """LLM calls read_kernel_source then submit_ncu_config — ncu_config is set."""
    from kernelopt.agents.profiler.graph import profiler_subgraph
    from kernelopt.agents.profiler.tools import ProfilerToolServer

    kernel_file = tmp_path / "softmax.py"
    kernel_file.write_text(TRITON_SOURCE)
    server = ProfilerToolServer(str(kernel_file), TRITON_SOURCE)

    mock_llm = _make_mock_llm([
        [{"name": "read_kernel_source", "args": {}, "id": "tc1"}],
        [{"name": "submit_ncu_config", "args": VALID_CONFIG_ARGS, "id": "tc2"}],
    ])

    config = RunnableConfig(configurable={
        "tool_server": server,
        "llm": mock_llm,
        "llm_config": {},
    })

    result = profiler_subgraph.invoke(
        {"messages": [HumanMessage(content="go")], "ncu_config": None},
        config=config,
    )

    assert result["ncu_config"] is not None
    assert result["ncu_config"]["kernel_name"] == "softmax_kernel"


def test_profiler_subgraph_no_submit_returns_none_ncu_config(tmp_path):
    """LLM never calls submit_ncu_config — graph exits with ncu_config=None."""
    from kernelopt.agents.profiler.graph import profiler_subgraph
    from kernelopt.agents.profiler.tools import ProfilerToolServer

    kernel_file = tmp_path / "softmax.py"
    kernel_file.write_text(TRITON_SOURCE)
    server = ProfilerToolServer(str(kernel_file), TRITON_SOURCE)

    mock_llm = _make_mock_llm([[]])  # no tool calls → route_from_agent → END

    config = RunnableConfig(configurable={
        "tool_server": server,
        "llm": mock_llm,
        "llm_config": {},
    })

    result = profiler_subgraph.invoke(
        {"messages": [HumanMessage(content="go")], "ncu_config": None},
        config=config,
    )

    assert result.get("ncu_config") is None
