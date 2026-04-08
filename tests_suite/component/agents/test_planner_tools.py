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
