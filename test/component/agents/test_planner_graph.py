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


# --- run_planner_agent integration ---

from unittest.mock import MagicMock
from kernelopt.agents.planner import run_planner_agent


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
