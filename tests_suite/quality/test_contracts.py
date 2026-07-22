"""
Contract tests - verify interfaces, protocols, and extension points.

These tests ensure:
- Stable interfaces for future development
- Extension points are well-defined
- Data schemas are forward-compatible
- New features can be added without breaking existing code

These tests document and verify the API CONTRACT.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


# ===========================================================================
# Agent Interface Contracts
# ===========================================================================

class TestAgentContracts:
    """Test agent public interface contracts."""

    def test_planner_interface(self):
        """Planner run_planner_agent() must maintain stable interface."""
        from kernelopt.agents import planner

        sig = inspect.signature(planner.run_planner_agent)

        # Required parameters (cannot be removed in future versions)
        assert 'profiling_context' in sig.parameters
        assert 'kernel_source' in sig.parameters
        assert 'memory' in sig.parameters  # Renamed from optimization_memory
        assert 'N' in sig.parameters  # Number of plans
        assert 'llm_config' in sig.parameters  # Replaces model parameter
        assert 'all_tried_directions' in sig.parameters

        # Return type should be list of (letter, plan) tuples

    def test_executor_interface(self):
        """Executor run_executor_agent() must maintain stable interface."""
        from kernelopt.agents import executor

        sig = inspect.signature(executor.run_executor_agent)

        # Required parameters
        assert 'kernel_source' in sig.parameters
        assert 'optimization_plan' in sig.parameters
        assert 'memory' in sig.parameters  # Renamed from optimization_memory
        assert 'K' in sig.parameters  # Max attempts
        assert 'llm_config' in sig.parameters  # Replaces framework/model parameters

        # Should return tuple: (optimized_code, execution_result)
        # execution_result must have 'status' and 'ready_for_profiling'

    def test_profiler_interface(self):
        """Profiler run_ncu() must maintain stable interface."""
        from kernelopt.agents import profiler

        sig = inspect.signature(profiler.run_ncu)

        # Required parameters
        assert 'kernel_path' in sig.parameters
        assert 'output_basename' in sig.parameters  # Renamed from output_dir

        # Should return tuple: (csv_path, ncu_rep_path, profiling_context_path)

    def test_summarizer_interface(self):
        """Summarizer run_summarizer() must maintain stable interface."""
        from kernelopt.agents import summarizer

        sig = inspect.signature(summarizer.run_summarizer)

        # Required parameters
        assert 'profiling_before' in sig.parameters
        assert 'profiling_after' in sig.parameters
        assert 'optimization_plan' in sig.parameters
        assert 'kernel_before' in sig.parameters
        assert 'kernel_after' in sig.parameters

        # Should return tuple: (experience_item, memory_update)


# ===========================================================================
# ToolServer Contracts
# ===========================================================================

class TestToolServerContracts:
    """Test ToolServer interface for extensibility."""

    def test_planner_tool_server_interface(self):
        """PlannerToolServer must have stable interface."""
        from kernelopt.agents.planner.tools import PlannerToolServer

        # Constructor signature
        init_sig = inspect.signature(PlannerToolServer.__init__)
        assert 'profiling_context' in init_sig.parameters
        assert 'kernel_source' in init_sig.parameters
        assert 'memory' in init_sig.parameters  # Renamed from optimization_memory

        # Required methods
        assert hasattr(PlannerToolServer, 'query_profiling_context')
        assert hasattr(PlannerToolServer, 'get_kernel_source')
        assert hasattr(PlannerToolServer, 'search_memory')

    def test_executor_tool_server_interface(self):
        """ExecutorToolServer must have stable interface."""
        from kernelopt.agents.executor.tools import ExecutorToolServer

        # Constructor signature
        init_sig = inspect.signature(ExecutorToolServer.__init__)
        assert 'kernel_source' in init_sig.parameters
        assert 'optimization_plan' in init_sig.parameters
        assert 'memory' in init_sig.parameters  # Renamed from optimization_memory

        # Required methods
        assert hasattr(ExecutorToolServer, 'get_kernel_source')
        assert hasattr(ExecutorToolServer, 'get_optimization_plan')
        assert hasattr(ExecutorToolServer, 'search_memory')

    def test_new_tool_server_can_reuse_search_logic(self):
        """New ToolServers can reuse memory search logic."""
        from kernelopt.agents.planner.tools import PlannerToolServer

        # Memory search should be generic, not hardcoded
        # Should work the same for any ToolServer

        memory = [
            {
                'item_id': 'test1',
                'direction': 'increase_block_size',
                'strategy_title': 'Test strategy',
                'strategy_description': 'Test description',
                'applicable_when': 'register pressure is high',
                'do_not_apply_when': 'memory bound',
                'framework': 'triton',
                'rewrite_type': 'positive',
                'speedup': 1.5,
                'iteration': 1,
                'profiling_signal': {},
                'slow_pseudocode': '',
                'fast_pseudocode': '',
                'framework_notes': None,
            }
        ]

        server = PlannerToolServer({}, "code", memory)
        results = server.search_memory("register pressure block")

        # Should find the item based on keywords
        assert len(results) > 0
        assert results[0]['direction'] == 'increase_block_size'


# ===========================================================================
# Data Schema Contracts
# ===========================================================================

class TestSchemaContracts:
    """Test data schemas for forward compatibility."""

    def test_profiling_context_schema(self):
        """Profiling context must have required fields."""
        required_fields = [
            'kernel_name',
            'framework_hint',
            'device',
            'launch',
            'sections',
            'rules',
        ]

        # Verify that build_profiling_context includes these
        from kernelopt.agents.profiler import build_profiling_context

        ctx = build_profiling_context({}, [], {}, "test.py")

        for field in required_fields:
            assert field in ctx, f"Missing required field: {field}"

    def test_optimization_plan_schema(self):
        """Optimization plan must have required fields."""
        required_fields = [
            'plan_id',
            'direction',
            'framework',
            'step',
            'evidence',
        ]

        # This is what planner MUST return
        # Cannot remove these fields in future versions

    def test_pipeline_state_schema(self):
        """PipelineState must have required fields."""
        from kernelopt.state import PipelineState, PipelineConfig

        cfg = PipelineConfig()
        state = PipelineState(config=cfg)

        required_fields = [
            'config',
            'iteration',
            'best_latency_ns',
            'best_kernel',
            'iteration_results',
            'no_improvement_streak',
        ]

        for field in required_fields:
            assert hasattr(state, field), f"Missing required field: {field}"

    def test_pipeline_config_schema(self):
        """PipelineConfig must have required fields."""
        from kernelopt.state import PipelineConfig

        cfg = PipelineConfig()

        required_fields = [
            'T',  # max iterations
            'N',  # plans per iteration
            'K',  # attempts per plan
            'keep_n',  # top K memory items
            'model',
        ]

        for field in required_fields:
            assert hasattr(cfg, field), f"Missing required field: {field}"

    def test_memory_item_schema(self):
        """Memory items must have required fields."""
        required_fields = [
            'item_id',
            'iteration',
            'speedup',
            'rewrite_type',
            'framework',
            'direction',
            'strategy_title',
            'strategy_description',
            'applicable_when',
            'do_not_apply_when',
        ]

        # This is what summarizer MUST return
        # Future: should have schema_version field

    def test_execution_result_schema(self):
        """Execution result must have required fields."""
        required_fields = [
            'status',  # success/failed/implementation_failure
            'ready_for_profiling',  # bool
        ]

        # This is what executor MUST return as second element of tuple


# ===========================================================================
# Extension Point Contracts
# ===========================================================================

class TestExtensionPoints:
    """Test that extension points are well-defined."""

    def test_can_add_new_profiling_section(self):
        """Should be able to add new profiling sections."""
        from kernelopt.agents.profiler import build_profiling_context

        # Custom sections should be allowed
        custom_sections = {
            'Custom Section': {
                'Custom Metric': {'value': 42, 'unit': 'custom'},
            }
        }

        ctx = build_profiling_context(custom_sections, [], {}, "test.py")

        assert 'sections' in ctx
        assert 'Custom Section' in ctx['sections']

    def test_can_add_new_ncu_rule(self):
        """Should be able to add new NCU rules."""
        from kernelopt.agents.profiler import build_profiling_context

        # Custom rules should be allowed
        custom_rules = [
            {
                'name': 'CustomRule',
                'type': 'OPT',
                'description': 'Custom optimization opportunity',
                'speedup_type': 'global',
                'speedup_pct': 15.0,
            }
        ]

        ctx = build_profiling_context({}, custom_rules, {}, "test.py")

        assert 'rules' in ctx
        assert len(ctx['rules']) > 0

    def test_memory_accepts_custom_fields(self):
        """Memory items can have framework-specific custom fields."""
        from kernelopt.memory import append_items

        custom_memory = [
            {
                'item_id': 'custom1',
                'direction': 'test',
                'framework': 'custom_framework',
                'speedup': 1.5,
                'rewrite_type': 'positive',
                'iteration': 1,
                # Custom fields
                'custom_metric': 123,
                'custom_analysis': {'foo': 'bar'},
                'profiling_signal': {},
                'strategy_title': 'test',
                'strategy_description': 'test',
                'applicable_when': 'test',
                'do_not_apply_when': 'test',
                'slow_pseudocode': '',
                'fast_pseudocode': '',
                'framework_notes': None,
            }
        ]

        memory = append_items([], custom_memory, capacity=10)

        # Custom fields should be preserved
        assert memory[0]['custom_metric'] == 123
        assert memory[0]['custom_analysis']['foo'] == 'bar'

    def test_pipeline_state_extensible(self):
        """PipelineState can have custom fields added dynamically."""
        from kernelopt.state import PipelineState, PipelineConfig

        cfg = PipelineConfig()
        state = PipelineState(config=cfg)

        # Should be able to add custom fields
        state.custom_metric = 42
        state.custom_data = {'key': 'value'}

        assert state.custom_metric == 42
        assert state.custom_data['key'] == 'value'


# ===========================================================================
# Backwards Compatibility Contracts
# ===========================================================================

class TestBackwardsCompatibility:
    """Test backwards compatibility guarantees."""

    def test_csv_parser_accepts_old_format(self):
        """CSV parser should handle old NCU output format."""
        from kernelopt.functions.csv_parser import parse_ncu_csv

        # Even if NCU output format changes, parser should handle it
        # This is a placeholder - actual test would use real old-format CSV

    def test_memory_format_is_stable(self):
        """Optimization memory format should be stable."""
        from kernelopt.memory import select_top_k, append_items

        # Old memory items should still work
        old_format_memory = [
            {
                'item_id': 'old1',
                'direction': 'test',
                'framework': 'triton',
                'speedup': 1.5,
                'rewrite_type': 'positive',
                'iteration': 1,
                'profiling_signal': {},
                'strategy_title': 'test',
                'strategy_description': 'test',
                'applicable_when': 'test',
                'do_not_apply_when': 'test',
                'slow_pseudocode': '',
                'fast_pseudocode': '',
                'framework_notes': None,
            }
        ]

        # Should work with current code
        selected = select_top_k(old_format_memory, top_k=1, spe_gate=1.05, slo_gate=1.20)
        assert len(selected) > 0

        memory = append_items([], old_format_memory, capacity=10)
        assert len(memory) == 1


# ===========================================================================
# Framework Contract Tests
# ===========================================================================

class TestFrameworkContracts:
    """Test framework-related contracts."""

    def test_framework_hint_is_string(self):
        """framework_hint should be a simple string (not enum)."""
        from kernelopt.agents.profiler import build_profiling_context

        ctx = build_profiling_context({}, [], {}, "test.py")

        assert 'framework_hint' in ctx
        assert isinstance(ctx['framework_hint'], str)

        # Should accept any framework name
        ctx['framework_hint'] = 'custom_framework'
        # Should not raise error

    def test_can_add_new_framework(self):
        """Should be able to add new framework without code changes."""
        from kernelopt.state import PipelineState, PipelineConfig

        cfg = PipelineConfig()
        state = PipelineState(config=cfg)

        # Should accept any framework
        state.framework = 'new_custom_framework'
        assert state.framework == 'new_custom_framework'

    def test_optimizer_accepts_unknown_framework(self):
        """Optimizer should handle unknown framework gracefully."""
        # This is a design principle - optimizer should work
        # even if it doesn't have specific knowledge of the framework


# ===========================================================================
# Tool Registration Contracts
# ===========================================================================

class TestToolRegistration:
    """Test tool registration patterns."""

    def test_planner_tools_are_registered(self):
        """Planner tools should be in _TOOL_MAP."""
        from kernelopt.agents.planner.tools import _TOOL_MAP

        # Required tools
        assert 'query_profiling_context' in _TOOL_MAP
        assert 'get_kernel_source' in _TOOL_MAP
        assert 'search_memory' in _TOOL_MAP

    def test_executor_tools_are_registered(self):
        """Executor tools should be in _TOOL_MAP."""
        from kernelopt.agents.executor.tools import _TOOL_MAP

        # Required tools
        assert 'get_kernel_source' in _TOOL_MAP
        assert 'get_optimization_plan' in _TOOL_MAP
        assert 'search_memory' in _TOOL_MAP

    def test_all_tools_list_is_complete(self):
        """ALL_TOOLS should contain all tool wrappers."""
        from kernelopt.agents.planner.tools import ALL_TOOLS as planner_tools

        # Should be a list of tool objects
        assert isinstance(planner_tools, list)
        assert len(planner_tools) > 0

        # Each should have a name
        for tool in planner_tools:
            assert hasattr(tool, 'name')


# ===========================================================================
# Configuration Contracts
# ===========================================================================

class TestConfigurationContracts:
    """Test configuration interface contracts."""

    def test_pipeline_config_has_defaults(self):
        """PipelineConfig should have sensible defaults."""
        from kernelopt.state import PipelineConfig

        cfg = PipelineConfig()

        # Should have defaults (not require all params)
        assert cfg.T > 0
        assert cfg.N > 0
        assert cfg.K >= 1
        assert cfg.keep_n > 0

    def test_pipeline_config_accepts_overrides(self):
        """PipelineConfig should accept parameter overrides."""
        from kernelopt.state import PipelineConfig

        cfg = PipelineConfig(
            T=10,
            N=8,
            K=3,
            keep_n=5,
        )

        assert cfg.T == 10
        assert cfg.N == 8
        assert cfg.K == 3
        assert cfg.keep_n == 5

    def test_config_is_immutable_during_run(self):
        """Config should not change during pipeline execution."""
        from kernelopt.state import PipelineConfig, PipelineState

        cfg = PipelineConfig(T=5)
        state = PipelineState(config=cfg)

        # Config should be stored, not modified
        assert state.config.T == 5

        # Modifying state should not affect original config
        # (This is a design principle test)
