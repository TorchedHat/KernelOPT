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
