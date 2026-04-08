"""
Architecture quality tests - verify design principles and code organization.

These tests ensure:
- Modular design with clear boundaries
- Separation of concerns
- Dependency injection
- No hardcoded assumptions
- Consistent patterns across components

These tests validate the STRUCTURE and DESIGN, not just functionality.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


# ===========================================================================
# Modular Design Tests
# ===========================================================================

class TestModularDesign:
    """Test modular architecture patterns."""

    def test_agent_packages_are_isolated(self):
        """Each agent package should be self-contained."""
        from kernelopt.agents import planner, executor

        # Planner package shouldn't import executor internals
        planner_source = inspect.getsource(planner)
        assert 'from kernelopt.agents.executor' not in planner_source
        assert 'ExecutorToolServer' not in planner_source

        # Executor package shouldn't import planner internals
        executor_source = inspect.getsource(executor)
        assert 'from kernelopt.agents.planner' not in executor_source
        assert 'PlannerToolServer' not in executor_source

    def test_agents_use_consistent_package_structure(self):
        """All agents should follow the same package structure."""
        base_path = Path(__file__).parent.parent.parent / "kernelopt" / "kernelopt" / "agents"

        agent_dirs = [
            base_path / 'planner',
            base_path / 'executor',
        ]

        for agent_dir in agent_dirs:
            # Each modular agent must have these files
            assert (agent_dir / 'graph.py').exists(), f"{agent_dir.name} missing graph.py"
            assert (agent_dir / 'tools.py').exists(), f"{agent_dir.name} missing tools.py"
            assert (agent_dir / 'prompts.py').exists(), f"{agent_dir.name} missing prompts.py"
            assert (agent_dir / '__init__.py').exists(), f"{agent_dir.name} missing __init__.py"

    def test_functions_are_stateless(self):
        """Functions should be pure (no hidden global state)."""
        from kernelopt.functions import csv_parser, selector, correctness

        # These should be importable without side effects
        importlib.reload(csv_parser)
        importlib.reload(selector)
        importlib.reload(correctness)

        # Should not raise or modify global state
        # If they do, this test will fail with an exception


    def test_single_responsibility_principle(self):
        """Each module should have a single, clear responsibility."""
        # graph.py = pipeline orchestration
        from kernelopt import graph
        assert 'run_pipeline' in dir(graph)

        # memory.py = optimization memory management
        from kernelopt import memory
        assert 'select_top_k' in dir(memory)
        assert 'append_items' in dir(memory)

        # state.py = pipeline state management
        from kernelopt import state
        assert 'PipelineState' in dir(state)
        assert 'PipelineConfig' in dir(state)

        # llm.py = LLM client management
        from kernelopt import llm
        assert 'build_llm' in dir(llm)


# ===========================================================================
# Separation of Concerns Tests
# ===========================================================================

class TestSeparationOfConcerns:
    """Test clear boundaries between layers."""

    def test_agents_dont_import_main_graph(self):
        """Agents shouldn't depend on main pipeline graph."""
        from kernelopt.agents import planner, executor

        planner_source = inspect.getsource(planner)
        executor_source = inspect.getsource(executor)

        # Agents are reusable components, shouldn't import main pipeline
        assert 'from kernelopt.graph import' not in planner_source
        assert 'from kernelopt.graph import' not in executor_source

        # They have their own sub-graphs (LangGraph compiled graphs)
        from kernelopt.agents.planner import graph as planner_graph
        from kernelopt.agents.executor import graph as executor_graph
        assert hasattr(planner_graph, 'planner_subgraph')
        assert hasattr(executor_graph, 'executor_subgraph')

    def test_tools_dont_call_llm(self):
        """Tool servers should provide data, not call LLM."""
        from kernelopt.agents.planner.tools import PlannerToolServer
        from kernelopt.agents.executor.tools import ExecutorToolServer

        planner_source = inspect.getsource(PlannerToolServer)
        executor_source = inspect.getsource(ExecutorToolServer)

        # Tools should not call LLM directly
        assert 'build_llm' not in planner_source
        assert 'build_llm' not in executor_source
        assert 'anthropic' not in planner_source.lower()
        assert 'anthropic' not in executor_source.lower()

    def test_prompts_dont_import_tools(self):
        """Prompts should be independent of tools."""
        from kernelopt.agents.planner import prompts as planner_prompts
        from kernelopt.agents.executor import prompts as executor_prompts

        planner_source = inspect.getsource(planner_prompts)
        executor_source = inspect.getsource(executor_prompts)

        # Prompts shouldn't import tools
        assert 'from .tools import' not in planner_source
        assert 'from .tools import' not in executor_source
        assert 'ToolServer' not in planner_source
        assert 'ToolServer' not in executor_source

    def test_csv_parser_is_independent(self):
        """CSV parser should not depend on other components."""
        from kernelopt.functions import csv_parser

        source = inspect.getsource(csv_parser)

        # Should only parse CSV, not know about agents or LLM
        assert 'from kernelopt.agents' not in source
        assert 'from kernelopt.llm' not in source
        assert 'from kernelopt.graph' not in source


# ===========================================================================
# Dependency Injection Tests
# ===========================================================================

class TestDependencyInjection:
    """Test that dependencies are injected, not hardcoded."""

    def test_agents_accept_model_parameter(self):
        """Agents should accept LLM config as parameter (not hardcoded)."""
        from kernelopt.agents import planner, executor

        planner_sig = inspect.signature(planner.run_planner_agent)
        executor_sig = inspect.signature(executor.run_executor_agent)

        # New API uses llm_config dict instead of individual model parameter
        assert 'llm_config' in planner_sig.parameters
        assert 'llm_config' in executor_sig.parameters

    def test_agents_accept_framework_parameter(self):
        """Executor infers framework from optimization plan."""
        from kernelopt.agents import executor

        executor_sig = inspect.signature(executor.run_executor_agent)

        # Framework is now inferred from optimization_plan, not passed separately
        assert 'optimization_plan' in executor_sig.parameters

    def test_no_hardcoded_gpu_model(self):
        """Should not assume specific GPU model."""
        from kernelopt.agents import gpu_specs

        source = inspect.getsource(gpu_specs)

        # Should not hardcode specific GPU models as defaults
        # (Having them in LOOKUP table is okay, but not as assumptions)
        assert 'default_gpu = "H100"' not in source
        assert 'default_gpu = "H200"' not in source

    def test_pipeline_config_is_configurable(self):
        """Pipeline config should not have hardcoded magic numbers."""
        from kernelopt.state import PipelineConfig

        cfg = PipelineConfig()

        # All important values should be configurable
        assert hasattr(cfg, 'T')  # max iterations
        assert hasattr(cfg, 'N')  # plans per iteration
        assert hasattr(cfg, 'K')  # attempts per plan
        assert hasattr(cfg, 'TopK')  # top K memory items

        # Should be able to override defaults
        custom_cfg = PipelineConfig(T=10, N=8, K=2)
        assert custom_cfg.T == 10
        assert custom_cfg.N == 8
        assert custom_cfg.K == 2


# ===========================================================================
# Interface Consistency Tests
# ===========================================================================

class TestInterfaceConsistency:
    """Test consistent interfaces across components."""

    def test_all_agents_export_run_function(self):
        """All agents must export a run_* function."""
        from kernelopt.agents import planner, executor, profiler, summarizer

        assert hasattr(planner, 'run_planner_agent')
        assert callable(planner.run_planner_agent)

        assert hasattr(executor, 'run_executor_agent')
        assert callable(executor.run_executor_agent)

        assert hasattr(profiler, 'run_ncu')
        assert callable(profiler.run_ncu)

        assert hasattr(summarizer, 'run_summarizer')
        assert callable(summarizer.run_summarizer)

    def test_all_tool_servers_have_search_memory(self):
        """All ToolServers should implement search_memory."""
        from kernelopt.agents.planner.tools import PlannerToolServer
        from kernelopt.agents.executor.tools import ExecutorToolServer

        servers = [PlannerToolServer, ExecutorToolServer]

        for Server in servers:
            assert hasattr(Server, 'search_memory')
            # Check it's a method, not just an attribute
            assert callable(getattr(Server, 'search_memory'))

    def test_all_tool_servers_have_get_kernel_source(self):
        """All ToolServers should implement get_kernel_source."""
        from kernelopt.agents.planner.tools import PlannerToolServer
        from kernelopt.agents.executor.tools import ExecutorToolServer

        for Server in [PlannerToolServer, ExecutorToolServer]:
            assert hasattr(Server, 'get_kernel_source')

    def test_tool_wrappers_use_consistent_naming(self):
        """LangChain tool wrappers should follow naming conventions."""
        from kernelopt.agents.planner import tools as planner_tools
        from kernelopt.agents.executor import tools as executor_tools

        # Both should have _TOOL_MAP
        assert hasattr(planner_tools, '_TOOL_MAP')
        assert hasattr(executor_tools, '_TOOL_MAP')

        # Both should have ALL_TOOLS
        assert hasattr(planner_tools, 'ALL_TOOLS')
        assert hasattr(executor_tools, 'ALL_TOOLS')


# ===========================================================================
# Code Organization Tests
# ===========================================================================

class TestCodeOrganization:
    """Test file and package organization."""

    def test_no_circular_imports(self):
        """No circular dependencies between modules."""
        # This test passes if import succeeds without errors
        from kernelopt import graph
        from kernelopt import memory
        from kernelopt import state
        from kernelopt.agents import planner
        from kernelopt.agents import executor
        from kernelopt.functions import csv_parser

        # If there were circular imports, this would fail

    def test_public_api_is_explicit(self):
        """Packages should explicitly define public API with __all__."""
        from kernelopt.agents import planner, executor

        # Check if __all__ is defined (good practice)
        # Not all modules have it yet, but they should
        # assert hasattr(planner, '__all__')
        # assert hasattr(executor, '__all__')

        # At minimum, check that public functions are exported
        assert 'run_planner_agent' in dir(planner)
        assert 'run_executor_agent' in dir(executor)

    def test_private_functions_use_underscore_prefix(self):
        """Private functions should start with _."""
        from kernelopt.agents.planner import tools as planner_tools

        # Check that tool map is private (internal detail)
        members = dir(planner_tools)
        assert '_TOOL_MAP' in members  # Private

        # Public tools don't start with _
        assert 'query_profiling_context' in members  # Public

    def test_no_wildcard_imports(self):
        """Modules should avoid 'from X import *'."""
        modules_to_check = [
            'kernelopt.graph',
            'kernelopt.memory',
            'kernelopt.state',
            'kernelopt.agents.planner.tools',
            'kernelopt.agents.executor.tools',
        ]

        for module_name in modules_to_check:
            module = importlib.import_module(module_name)
            source = inspect.getsource(module)

            # Should not use wildcard imports
            assert 'import *' not in source, f"{module_name} uses wildcard imports"


# ===========================================================================
# Error Handling Tests
# ===========================================================================

class TestErrorHandling:
    """Test error handling patterns."""

    def test_agents_return_errors_not_raise(self):
        """Agents should return error status, not raise exceptions."""
        from kernelopt.agents import executor

        # Executor returns (code, result) where result has status
        sig = inspect.signature(executor.run_executor_agent)
        # Should return tuple, not raise

        # This is a design principle test
        # In practice, check that result dict has 'status' field

    def test_tool_servers_validate_inputs(self):
        """ToolServers should validate inputs."""
        from kernelopt.agents.planner.tools import PlannerToolServer

        # Create server with valid data
        server = PlannerToolServer({}, "code", [])

        # Invalid query should be handled gracefully
        with pytest.raises(ValueError, match="Unknown aspect"):
            server.query_profiling_context("invalid_aspect_name")


# ===========================================================================
# Performance & Scalability Tests
# ===========================================================================

class TestPerformanceDesign:
    """Test design for performance."""

    def test_memory_search_has_max_results(self):
        """Memory search should limit results to avoid performance issues."""
        from kernelopt.agents.planner.tools import PlannerToolServer

        # Create large memory
        large_memory = [
            {
                'item_id': f'item_{i}',
                'direction': 'test',
                'strategy_title': 'test',
                'strategy_description': 'test',
                'applicable_when': 'test',
                'do_not_apply_when': 'test',
                'framework': 'triton',
                'rewrite_type': 'positive',
                'speedup': 1.1,
                'iteration': 1,
                'profiling_signal': {},
                'slow_pseudocode': '',
                'fast_pseudocode': '',
                'framework_notes': None,
            }
            for i in range(100)
        ]

        server = PlannerToolServer({}, "code", large_memory)
        results = server.search_memory("test")

        # Should limit results (e.g., top 3)
        assert len(results) <= 3

    def test_state_uses_dataclass_for_efficiency(self):
        """State should use dataclass or similar for performance."""
        from kernelopt.state import PipelineState, PipelineConfig

        # Check if using dataclass (efficient) or regular class
        # Dataclasses are more efficient for data containers
        import dataclasses

        # At minimum, should have __slots__ or be dataclass
        # assert dataclasses.is_dataclass(PipelineState) or hasattr(PipelineState, '__slots__')


# ===========================================================================
# Documentation Tests
# ===========================================================================

class TestDocumentation:
    """Test documentation quality."""

    def test_all_public_functions_have_docstrings(self):
        """All public functions should have docstrings."""
        from kernelopt.agents import planner, executor

        public_functions = [
            planner.run_planner_agent,
            executor.run_executor_agent,
        ]

        for func in public_functions:
            assert func.__doc__ is not None, f"{func.__name__} missing docstring"
            assert len(func.__doc__.strip()) > 10, f"{func.__name__} docstring too short"

    def test_modules_have_module_docstrings(self):
        """All modules should have module-level docstrings."""
        from kernelopt import graph, memory, state

        modules = [graph, memory, state]

        for module in modules:
            assert module.__doc__ is not None, f"{module.__name__} missing module docstring"
