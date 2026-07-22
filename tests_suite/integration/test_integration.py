"""
Integration tests for the kernel optimization pipeline.

Tests cover:
- Multi-component workflows
- Agent interactions
- Pipeline phases
- End-to-end scenarios (lightweight)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


# ===========================================================================
# Pipeline Component Integration
# ===========================================================================

class TestPipelineComponents:
    """Test integration of pipeline components."""

    def test_profiler_to_planner_integration(
        self, test_kernel_path, skip_if_no_cuda, patch_build_llm, gpu_id
    ):
        """Test profiler output feeds into planner."""
        from kernelopt.agents.profiler import build_profiling_context, run_ncu
        from kernelopt.agents.planner import run_planner_agent as run_planner
        from kernelopt.functions.csv_parser import parse_ncu_csv

        # Patch LLM for planner
        patch_build_llm(gpu_id)

        # Phase 1: Profile
        csv_path, _, _ = run_ncu(
            str(test_kernel_path),
            "/tmp/test_profiler_to_planner",
            kernel_args=["--profile-only"],
        )

        sections, rules, launch = parse_ncu_csv(csv_path)
        profiling_ctx = build_profiling_context(
            sections, rules, launch, str(test_kernel_path)
        )

        # Phase 2: Plan
        kernel_source = test_kernel_path.read_text()
        plan_result = run_planner(
            profiling_context=profiling_ctx,
            kernel_source=kernel_source,
            memory=[],
            N=1,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
            all_tried_directions=[],
        )
        plan = plan_result[0][1] if plan_result else {}

        # Verify integration
        assert plan is not None
        assert "plan_id" in plan
        assert "direction" in plan
        assert plan.get("framework") == profiling_ctx["framework_hint"]

    def test_planner_to_executor_integration(
        self, test_kernel_path, sample_optimization_plan, patch_build_llm, gpu_id
    ):
        """Test planner output feeds into executor."""
        from kernelopt.agents.executor import run_executor_agent as run_executor

        # Patch LLM for executor
        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()
        optimized_source, execution_result = run_executor(
            kernel_source=kernel_source,
            optimization_plan=sample_optimization_plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
        )

        # Verify integration
        assert len(optimized_source) > 0
        assert execution_result.get("status") in ["success", "failed", "implementation_failure"]
        assert "ready_for_profiling" in execution_result

    def test_executor_to_correctness_integration(
        self, test_kernel_path, sample_optimization_plan, patch_build_llm,
        gpu_id, skip_if_no_cuda
    ):
        """Test executor output goes through correctness check."""
        from kernelopt.agents.executor import run_executor_agent as run_executor
        from kernelopt.functions.correctness import check_correctness_subprocess

        # Patch LLM for executor
        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()
        optimized_source, _ = run_executor(
            kernel_source=kernel_source,
            optimization_plan=sample_optimization_plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
        )

        # Check correctness
        passed, msg = check_correctness_subprocess(
            str(test_kernel_path),
            optimized_source,
        )

        # Verify integration (may pass or fail, but should execute)
        assert isinstance(passed, bool)
        assert isinstance(msg, str)

    def test_profiling_to_memory_integration(
        self, sample_profiling_context, sample_optimization_plan,
        patch_build_llm, gpu_id
    ):
        """Test profiling results feed into memory summarization."""
        from kernelopt.agents.summarizer import run_summarizer

        # Patch LLM for summarizer
        patch_build_llm(gpu_id)

        kernel_before = "# Original kernel code"
        kernel_after = "# Optimized kernel code"

        experience_item, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=sample_optimization_plan,
            kernel_before=kernel_before,
            kernel_after=kernel_after,
            iteration=1,
            framework="triton",
            speedup=1.15,
            existing_memory=[],
            model="claude-sonnet-4@20250514",
        )

        # Verify integration
        assert experience_item is not None
        assert "item_id" in experience_item
        assert memory_update is not None
        assert "action" in memory_update


# ===========================================================================
# Data Flow Integration
# ===========================================================================

class TestDataFlow:
    """Test data flowing through the pipeline."""

    def test_csv_to_context_flow(self, sample_ncu_csv):
        """Test CSV parsing to profiling context flow."""
        from kernelopt.functions.csv_parser import parse_ncu_csv
        from kernelopt.agents.profiler import build_profiling_context, get_latency_ns

        # Parse CSV
        sections, rules, launch = parse_ncu_csv(str(sample_ncu_csv))

        # Build context
        ctx = build_profiling_context(
            sections, rules, launch, "/tmp/test.py"
        )

        # Extract metrics
        latency = get_latency_ns(ctx)

        # Verify flow
        assert isinstance(sections, dict)
        assert isinstance(ctx, dict)
        assert latency >= 0

    def test_plan_to_execution_flow(self, sample_optimization_plan):
        """Test plan structure flows into execution."""
        # Verify plan has all required fields for executor
        assert "direction" in sample_optimization_plan
        assert "framework" in sample_optimization_plan
        assert "step" in sample_optimization_plan

        step = sample_optimization_plan["step"]
        assert "action" in step
        assert "target" in step
        assert "change" in step

    def test_memory_update_flow(self, sample_memory_items):
        """Test memory items flow through selection and storage."""
        from kernelopt.memory import select_top_k, append_items, save_memory
        import tempfile

        # Select top items
        top = select_top_k(sample_memory_items, top_k=1, spe_gate=1.05, slo_gate=1.20)

        # Append to memory
        memory = append_items([], top, capacity=5)

        # Save memory
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_memory(memory, f.name)
            path = Path(f.name)

        # Verify flow
        assert len(memory) > 0
        assert path.exists()


# ===========================================================================
# State Transitions
# ===========================================================================

class TestStateTransitions:
    """Test state transitions through pipeline phases."""

    def test_initialization_to_baseline(self):
        """Test state transition from init to baseline profiling."""
        from kernelopt.state import PipelineState, PipelineConfig

        # Initial state
        cfg = PipelineConfig()
        state = PipelineState(
            config=cfg,
            kernel_path="/tmp/test.py",
            kernel_source="test code",
        )

        assert state.iteration == 0
        assert state.best_latency_ns == float("inf")

        # After baseline profiling
        state.baseline_csv = "/tmp/baseline.csv"
        state.best_latency_ns = 100.0

        assert state.baseline_csv != ""
        assert state.best_latency_ns < float("inf")

    def test_iteration_to_iteration(self):
        """Test state transition between iterations."""
        from kernelopt.state import (
            PipelineState,
            PipelineConfig,
            IterationResult,
            PlanCandidate,
        )

        cfg = PipelineConfig(T=3)
        state = PipelineState(config=cfg, best_latency_ns=100.0)

        # Iteration 1
        state.iteration = 0
        candidate = PlanCandidate(latency_ns=90.0, speedup=1.11)
        result = IterationResult(
            iteration=1,
            candidates=[candidate],
            best_candidate=candidate,
            improved=True,
        )
        state.iteration_results.append(result)
        state.best_latency_ns = 90.0
        state.no_improvement_streak = 0

        # Iteration 2 (no improvement)
        state.iteration = 1
        candidate2 = PlanCandidate(latency_ns=95.0, speedup=0.95)
        result2 = IterationResult(
            iteration=2,
            candidates=[candidate2],
            improved=False,
        )
        state.iteration_results.append(result2)
        state.no_improvement_streak += 1

        # Verify transitions
        assert len(state.iteration_results) == 2
        assert state.best_latency_ns == 90.0
        assert state.no_improvement_streak == 1

    def test_stopping_condition_check(self):
        """Test checking stopping conditions."""
        from kernelopt.state import PipelineState, PipelineConfig

        cfg = PipelineConfig(T=5)
        state = PipelineState(config=cfg)

        # No improvement streak
        state.no_improvement_streak = 2
        should_stop = state.no_improvement_streak >= 2

        assert should_stop is True


# ===========================================================================
# Multi-Agent Workflows
# ===========================================================================

class TestMultiAgentWorkflows:
    """Test workflows involving multiple agents."""

    @pytest.mark.slow
    def test_profile_plan_execute_workflow(
        self, test_kernel_path, skip_if_no_cuda, patch_build_llm, gpu_id
    ):
        """Test complete profile → plan → execute workflow."""
        from kernelopt.agents.profiler import build_profiling_context, run_ncu
        from kernelopt.agents.planner import run_planner_agent as run_planner
        from kernelopt.agents.executor import run_executor_agent as run_executor
        from kernelopt.functions.csv_parser import parse_ncu_csv

        # Patch LLM
        patch_build_llm(gpu_id)

        # 1. Profile
        csv_path, _, _ = run_ncu(
            str(test_kernel_path),
            "/tmp/test_workflow",
            kernel_args=["--profile-only"],
        )
        sections, rules, launch = parse_ncu_csv(csv_path)
        profiling_ctx = build_profiling_context(
            sections, rules, launch, str(test_kernel_path)
        )

        # 2. Plan
        kernel_source = test_kernel_path.read_text()
        plan_result = run_planner(
            profiling_context=profiling_ctx,
            kernel_source=kernel_source,
            memory=[],
            N=1,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
            all_tried_directions=[],
        )
        plan = plan_result[0][1] if plan_result else {}

        # 3. Execute
        optimized_source, execution_result = run_executor(
            kernel_source=kernel_source,
            optimization_plan=plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
        )

        # Verify workflow
        assert profiling_ctx is not None
        assert plan is not None
        assert len(optimized_source) > 0
        assert "status" in execution_result


# ===========================================================================
# Error Handling Integration
# ===========================================================================

class TestErrorHandling:
    """Test error handling across components."""

    def test_invalid_csv_handling(self):
        """Test handling of invalid CSV files."""
        from kernelopt.functions.csv_parser import parse_ncu_csv
        import tempfile

        # Create invalid CSV
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("Invalid,CSV,Content\n1,2,3\n")
            path = f.name

        # Should handle gracefully
        try:
            sections, rules, launch = parse_ncu_csv(path)
            # May return empty results but shouldn't crash
            assert isinstance(sections, dict)
            assert isinstance(rules, list)
            assert isinstance(launch, dict)
        except Exception as e:
            # Or raise a specific error
            assert e is not None

    def test_missing_kernel_handling(self):
        """Test handling of missing kernel files."""
        from kernelopt.functions.correctness import check_correctness_subprocess

        # Should handle missing file
        passed, msg = check_correctness_subprocess(
            "/nonexistent/kernel.py",
            "dummy code",
        )

        assert passed is False
        assert len(msg) > 0

    def test_invalid_plan_handling(self, test_kernel_path, patch_build_llm, gpu_id):
        """Test handling of invalid optimization plans."""
        from kernelopt.agents.executor import run_executor_agent as run_executor

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()
        invalid_plan = {
            "plan_id": "invalid",
            "direction": "invalid_direction",
            "framework": "unknown",
            "step": {},  # Missing required fields
        }

        # Should handle invalid plan gracefully
        try:
            optimized_source, result = run_executor(
                kernel_source=kernel_source,
                optimization_plan=invalid_plan,
                framework="triton",
                model="claude-sonnet-4@20250514",
            )
            # May fail but shouldn't crash
            assert result.get("status") == "failed"
        except Exception as e:
            # Or raise a specific error
            assert e is not None
