"""
Unit tests for state management and schema validation.

Tests cover:
- PipelineState dataclass
- PipelineConfig dataclass
- PlanCandidate dataclass
- IterationResult dataclass
- Pydantic schema models
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


# ===========================================================================
# Tests for State Dataclasses
# ===========================================================================

class TestPipelineConfig:
    """Test suite for PipelineConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        from kernelopt.state import PipelineConfig

        cfg = PipelineConfig()

        assert cfg.T > 0
        assert cfg.N > 0
        assert cfg.K > 0
        assert cfg.keep_n > 0
        assert cfg.q_cap > 0
        assert cfg.spe_gate > 1.0
        assert cfg.slo_gate > 1.0

    def test_custom_values(self):
        """Test setting custom configuration values."""
        from kernelopt.state import PipelineConfig

        cfg = PipelineConfig(
            T=10,
            N=8,
            K=3,
            keep_n=5,
            model="test-model",
        )

        assert cfg.T == 10
        assert cfg.N == 8
        assert cfg.K == 3
        assert cfg.keep_n == 5
        assert cfg.model == "test-model"

    def test_kernel_args_list(self):
        """Test kernel_args is a list."""
        from kernelopt.state import PipelineConfig

        cfg = PipelineConfig(kernel_args=["--profile-only", "--m", "1024"])

        assert isinstance(cfg.kernel_args, list)
        assert len(cfg.kernel_args) == 3

    def test_ncu_extra_args(self):
        """Test NCU extra args configuration."""
        from kernelopt.state import PipelineConfig

        cfg = PipelineConfig(ncu_extra_args=["--metrics", "memory"])

        assert isinstance(cfg.ncu_extra_args, list)
        assert "--metrics" in cfg.ncu_extra_args


class TestPipelineState:
    """Test suite for PipelineState dataclass."""

    def test_initialization(self):
        """Test basic state initialization."""
        from kernelopt.state import PipelineState, PipelineConfig

        cfg = PipelineConfig()
        state = PipelineState(
            config=cfg,
            kernel_path="/tmp/test.py",
            kernel_source="test code",
        )

        assert state.config == cfg
        assert state.kernel_path == "/tmp/test.py"
        assert state.kernel_source == "test code"

    def test_default_values(self):
        """Test default state values."""
        from kernelopt.state import PipelineState

        state = PipelineState()

        assert state.iteration == 0
        assert state.no_improvement_streak == 0
        assert state.best_latency_ns == float("inf")
        assert state.framework == "auto"
        assert isinstance(state.optimization_memory, list)
        assert len(state.optimization_memory) == 0

    def test_iteration_tracking(self):
        """Test iteration tracking."""
        from kernelopt.state import PipelineState

        state = PipelineState()

        assert state.iteration == 0

        state.iteration = 5
        assert state.iteration == 5

    def test_no_improvement_streak(self):
        """Test no improvement streak tracking."""
        from kernelopt.state import PipelineState

        state = PipelineState()

        state.no_improvement_streak = 0
        state.no_improvement_streak += 1
        assert state.no_improvement_streak == 1

        # Reset on improvement
        state.no_improvement_streak = 0
        assert state.no_improvement_streak == 0

    def test_optimization_memory_updates(self):
        """Test updating optimization memory."""
        from kernelopt.state import PipelineState

        state = PipelineState()

        item1 = {"item_id": "001", "speedup": 1.25}
        item2 = {"item_id": "002", "speedup": 0.85}

        state.optimization_memory.append(item1)
        state.optimization_memory.append(item2)

        assert len(state.optimization_memory) == 2
        assert item1 in state.optimization_memory

    def test_iteration_results_tracking(self):
        """Test tracking iteration results."""
        from kernelopt.state import PipelineState, IterationResult

        state = PipelineState()

        result1 = IterationResult(iteration=1)
        result2 = IterationResult(iteration=2)

        state.iteration_results.append(result1)
        state.iteration_results.append(result2)

        assert len(state.iteration_results) == 2
        assert state.iteration_results[0].iteration == 1


class TestPlanCandidate:
    """Test suite for PlanCandidate dataclass."""

    def test_initialization(self):
        """Test plan candidate initialization."""
        from kernelopt.state import PlanCandidate

        candidate = PlanCandidate(
            plan_letter="A",
            direction="optimize_memory",
            latency_ns=100.0,
        )

        assert candidate.plan_letter == "A"
        assert candidate.direction == "optimize_memory"
        assert candidate.latency_ns == 100.0

    def test_default_values(self):
        """Test default values for plan candidate."""
        from kernelopt.state import PlanCandidate

        candidate = PlanCandidate()

        assert candidate.latency_ns == float("inf")
        assert candidate.speedup == 1.0
        assert candidate.correctness_passed is False
        assert isinstance(candidate.optimization_plan, dict)

    def test_speedup_calculation(self):
        """Test speedup tracking."""
        from kernelopt.state import PlanCandidate

        candidate = PlanCandidate(
            latency_ns=80.0,
            speedup=1.25,
        )

        assert candidate.speedup == 1.25

    def test_correctness_tracking(self):
        """Test correctness status tracking."""
        from kernelopt.state import PlanCandidate

        candidate = PlanCandidate(
            correctness_passed=True,
            correctness_message="All tests passed",
        )

        assert candidate.correctness_passed is True
        assert "passed" in candidate.correctness_message.lower()


class TestIterationResult:
    """Test suite for IterationResult dataclass."""

    def test_initialization(self):
        """Test iteration result initialization."""
        from kernelopt.state import IterationResult

        result = IterationResult(iteration=1)

        assert result.iteration == 1
        assert isinstance(result.candidates, list)
        assert result.improved is False

    def test_candidates_tracking(self):
        """Test tracking candidates in iteration."""
        from kernelopt.state import IterationResult, PlanCandidate

        result = IterationResult(iteration=1)

        c1 = PlanCandidate(plan_letter="A")
        c2 = PlanCandidate(plan_letter="B")

        result.candidates.append(c1)
        result.candidates.append(c2)

        assert len(result.candidates) == 2

    def test_best_candidate_tracking(self):
        """Test tracking best candidate."""
        from kernelopt.state import IterationResult, PlanCandidate

        result = IterationResult(iteration=1)
        best = PlanCandidate(plan_letter="A", latency_ns=80.0)

        result.best_candidate = best

        assert result.best_candidate is not None
        assert result.best_candidate.plan_letter == "A"

    def test_improvement_flag(self):
        """Test improvement flag."""
        from kernelopt.state import IterationResult

        result = IterationResult(iteration=1, improved=True)

        assert result.improved is True


# ===========================================================================
# Tests for Pydantic Schemas
# ===========================================================================

class TestSchemas:
    """Test suite for Pydantic schema models."""

    def test_metric_value_schema(self):
        """Test MetricValue schema."""
        from kernelopt.schemas import MetricValue

        metric = MetricValue(value=100.5, unit="GB/s")

        assert metric.value == 100.5
        assert metric.unit == "GB/s"

        # Test with string value
        metric2 = MetricValue(value="N/A", unit="")
        assert metric2.value == "N/A"

    def test_launch_config_schema(self):
        """Test LaunchConfig schema."""
        from kernelopt.schemas import LaunchConfig

        launch = LaunchConfig(
            block_size=256,
            grid_size=4096,
            sm_count=108,
        )

        assert launch.block_size == 256
        assert launch.grid_size == 4096
        assert launch.sm_count == 108

    def test_ncu_rule_schema(self):
        """Test NCURule schema."""
        from kernelopt.schemas import NCURule

        rule = NCURule(
            name="MemoryBound",
            type="OPT",
            description="Kernel is memory bound",
            speedup_type="global",
            speedup_pct=15.5,
        )

        assert rule.name == "MemoryBound"
        assert rule.type == "OPT"
        assert rule.speedup_pct == 15.5

    def test_profiling_context_schema(self):
        """Test ProfilingContext schema."""
        from kernelopt.schemas import ProfilingContext

        ctx = ProfilingContext(
            kernel_name="test_kernel",
            kernel_name_full="void test_kernel(float*)",
            framework_hint="cuda",
        )

        assert ctx.kernel_name == "test_kernel"
        assert ctx.framework_hint == "cuda"

    def test_plan_evidence_schema(self):
        """Test PlanEvidence schema."""
        from kernelopt.schemas import PlanEvidence

        evidence = PlanEvidence(
            ncu_rule="MemoryBound",
            ncu_rule_speedup_pct=15.5,
            supporting_metrics={"Memory Throughput": 850.5},
            reasoning="Improve memory access patterns",
        )

        assert evidence.ncu_rule == "MemoryBound"
        assert evidence.ncu_rule_speedup_pct == 15.5
        assert "Memory Throughput" in evidence.supporting_metrics

    def test_plan_step_schema(self):
        """Test PlanStep schema."""
        from kernelopt.schemas import PlanStep

        step = PlanStep(
            action="modify",
            target="memory access",
            change="Use coalesced accesses",
            predicted_effect="Improved bandwidth",
        )

        assert step.action == "modify"
        assert step.target == "memory access"

    def test_optimization_plan_schema(self, sample_optimization_plan):
        """Test OptimizationPlan schema validation."""
        from kernelopt.schemas import OptimizationPlan

        # This should validate the sample plan
        assert "plan_id" in sample_optimization_plan
        assert "direction" in sample_optimization_plan
        assert "framework" in sample_optimization_plan


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestStateIntegration:
    """Integration tests for state management."""

    def test_pipeline_state_workflow(self):
        """Test complete pipeline state workflow."""
        from kernelopt.state import (
            PipelineState,
            PipelineConfig,
            IterationResult,
            PlanCandidate,
        )

        # Setup
        cfg = PipelineConfig(T=3, N=4)
        state = PipelineState(
            config=cfg,
            kernel_path="/tmp/test.py",
            kernel_source="test",
            best_latency_ns=100.0,
        )

        # Iteration 1
        state.iteration = 0
        candidates = [
            PlanCandidate(
                plan_letter="A",
                latency_ns=90.0,
                speedup=1.11,
                correctness_passed=True,
            ),
            PlanCandidate(
                plan_letter="B",
                latency_ns=95.0,
                speedup=1.05,
                correctness_passed=True,
            ),
        ]

        best_c = min(candidates, key=lambda c: c.latency_ns)
        result = IterationResult(
            iteration=1,
            candidates=candidates,
            best_candidate=best_c,
            improved=best_c.latency_ns < state.best_latency_ns,
        )

        state.iteration_results.append(result)
        if result.improved:
            state.best_latency_ns = best_c.latency_ns

        # Verify
        assert len(state.iteration_results) == 1
        assert state.best_latency_ns == 90.0
        assert result.improved is True

    def test_config_to_state_integration(self):
        """Test creating state from config."""
        from kernelopt.state import PipelineState, PipelineConfig

        cfg = PipelineConfig(
            T=5,
            N=4,
            K=2,
            model="test-model",
            output_dir="/tmp/test_output",
        )

        state = PipelineState(
            config=cfg,
            kernel_path="/tmp/kernel.py",
        )

        assert state.config.T == 5
        assert state.config.N == 4
        assert state.config.model == "test-model"
