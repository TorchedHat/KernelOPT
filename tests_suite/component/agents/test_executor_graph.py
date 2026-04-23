# kernelopt/tests/agents/test_executor_graph.py
from __future__ import annotations

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
        # First submit: empty kernel (fails validation, attempt < K)
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


def test_run_executor_agent_passes_compile_error_in_initial_message(sample_memory):
    """compile_error kwarg is accepted without error and surfaces in initial message."""
    mock_llm = _make_mock_llm([
        [{"name": "submit_kernel", "args": {"kernel_source": TRITON_KERNEL, "change_summary": "fixed"}, "id": "tc1"}],
    ])
    kernel, result = run_executor_agent(
        kernel_source=TRITON_KERNEL,
        optimization_plan=SAMPLE_PLAN,
        memory=sample_memory,
        K=1,
        llm_config={"model": "unused", "api_key": None, "project_id": None, "region": None},
        llm_override=mock_llm,
        compile_error="kernel.cu(5): error: expected ';'",
    )
    assert result["status"] == "success"
    invoke_calls = mock_llm.invoke.call_args_list
    first_call_messages = invoke_calls[0][0][0]
    combined = " ".join(str(m.content) for m in first_call_messages)
    assert "expected ';'" in combined
