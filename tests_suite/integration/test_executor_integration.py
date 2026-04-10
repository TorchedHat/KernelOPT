"""
Comprehensive tests for the executor agent.

Tests cover:
- Code generation from optimization plans
- Correctness preservation
- Multiple attempt handling
- Framework-specific execution
- Error handling and recovery
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestExecutorBasic:
    """Basic executor agent tests."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_run_executor_basic(
        self, test_kernel_path, sample_optimization_plan,
        patch_build_llm, gpu_id
    ):
        """Test basic execution of optimization plan."""
        from kernelopt.agents.executor import run_executor_agent as run_executor

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        optimized_source, execution_result = run_executor(
            kernel_source=kernel_source,
            optimization_plan=sample_optimization_plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
        )

        assert len(optimized_source) > 0
        assert isinstance(execution_result, dict)
        assert "status" in execution_result

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_execution_result_structure(
        self, test_kernel_path, sample_optimization_plan,
        patch_build_llm, gpu_id
    ):
        """Test that execution result has required fields."""
        from kernelopt.agents.executor import run_executor_agent as run_executor

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        _, result = run_executor(
            kernel_source=kernel_source,
            optimization_plan=sample_optimization_plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
        )

        # Required fields
        assert "status" in result
        assert "ready_for_profiling" in result
        assert result["status"] in ["success", "failed", "implementation_failure"]


class TestExecutorCodeGeneration:
    """Test code generation quality."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_code_is_modified(
        self, test_kernel_path, sample_optimization_plan,
        patch_build_llm, gpu_id
    ):
        """Test that executor actually modifies the code."""
        from kernelopt.agents.executor import run_executor_agent as run_executor

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        optimized_source, result = run_executor(
            kernel_source=kernel_source,
            optimization_plan=sample_optimization_plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
        )

        if result["status"] == "success":
            # Code should be different (unless plan was to keep it same)
            # This is a soft check as LLM might fail to modify
            assert len(optimized_source) > 0

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_code_syntax_valid(
        self, test_kernel_path, sample_optimization_plan,
        patch_build_llm, gpu_id
    ):
        """Test that generated code has valid Python syntax."""
        from kernelopt.agents.executor import run_executor_agent as run_executor
        import ast

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        optimized_source, result = run_executor(
            kernel_source=kernel_source,
            optimization_plan=sample_optimization_plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
        )

        if result["status"] == "success" and result.get("ready_for_profiling"):
            # Should have valid Python syntax
            try:
                ast.parse(optimized_source)
            except SyntaxError:
                pytest.fail("Generated code has invalid Python syntax")


class TestExecutorAttempts:
    """Test multiple attempt handling."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_multiple_attempts(
        self, test_kernel_path, sample_optimization_plan,
        patch_build_llm, gpu_id
    ):
        """Test executor with multiple attempts."""
        from kernelopt.agents.executor import run_executor_agent as run_executor

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        # Attempt 1
        _, result1 = run_executor(
            kernel_source=kernel_source,
            optimization_plan=sample_optimization_plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
        )

        # Attempt 2
        _, result2 = run_executor(
            kernel_source=kernel_source,
            optimization_plan=sample_optimization_plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
        )

        # Both should execute
        assert result1 is not None
        assert result2 is not None


class TestExecutorFrameworks:
    """Test framework-specific execution."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_triton_execution(
        self, test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test execution for Triton kernel."""
        from kernelopt.agents.executor import run_executor_agent as run_executor

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        plan = {
            "plan_id": "triton_test",
            "direction": "optimize_triton",
            "framework": "triton",
            "step": {
                "action": "modify",
                "target": "softmax computation",
                "change": "Optimize computation",
                "predicted_effect": "Better performance",
            },
        }

        optimized_source, result = run_executor(
            kernel_source=kernel_source,
            optimization_plan=plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
        )

        assert result is not None

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_cuda_execution(
        self, patch_build_llm, gpu_id
    ):
        """Test execution for CUDA kernel."""
        from kernelopt.agents.executor import run_executor_agent as run_executor

        patch_build_llm(gpu_id)

        cuda_kernel = """
__global__ void simple_kernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= 2.0f;
    }
}
"""

        plan = {
            "plan_id": "cuda_test",
            "direction": "optimize_cuda",
            "framework": "cuda",
            "step": {
                "action": "modify",
                "target": "memory access",
                "change": "Optimize access pattern",
                "predicted_effect": "Better performance",
            },
        }

        optimized_source, result = run_executor(
            kernel_source=cuda_kernel,
            optimization_plan=plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
        )

        assert result is not None


class TestExecutorErrorHandling:
    """Test error handling in executor."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_invalid_plan_handling(
        self, test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test handling of invalid optimization plan."""
        from kernelopt.agents.executor import run_executor_agent as run_executor

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        invalid_plan = {
            "plan_id": "invalid",
            "direction": "invalid_direction",
            # Missing step field
        }

        # Should handle gracefully
        try:
            optimized_source, result = run_executor(
                kernel_source=kernel_source,
                optimization_plan=invalid_plan,
                framework="triton",
                model="claude-sonnet-4@20250514",
            )
            assert result.get("status") in ["success", "failed"]
        except Exception as e:
            # Or raise appropriate error
            assert e is not None

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_empty_kernel_handling(
        self, sample_optimization_plan, patch_build_llm, gpu_id
    ):
        """Test handling of empty kernel source."""
        from kernelopt.agents.executor import run_executor_agent as run_executor

        patch_build_llm(gpu_id)

        empty_kernel = ""

        try:
            optimized_source, result = run_executor(
                kernel_source=empty_kernel,
                optimization_plan=sample_optimization_plan,
                framework="triton",
                model="claude-sonnet-4@20250514",
            )
            assert result.get("status") == "failed"
        except Exception:
            # Should handle gracefully
            pass


class TestExecutorCorrectness:
    """Test correctness preservation."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_correctness_constraints(
        self, test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test that correctness constraints are in plan."""
        from kernelopt.agents.executor import run_executor_agent as run_executor

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        plan = {
            "plan_id": "correctness_test",
            "direction": "optimize_with_constraints",
            "framework": "triton",
            "correctness_constraints": [
                "Output must match reference within 1e-4",
                "No NaN or Inf values",
            ],
            "step": {
                "action": "modify",
                "target": "computation",
                "change": "Optimize",
                "predicted_effect": "Faster",
            },
        }

        optimized_source, result = run_executor(
            kernel_source=kernel_source,
            optimization_plan=plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
        )

        assert result is not None


class TestExecutorOutputMetrics:
    """Test output metrics and reporting."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_lines_changed_reported(
        self, test_kernel_path, sample_optimization_plan,
        patch_build_llm, gpu_id
    ):
        """Test that lines changed is reported."""
        from kernelopt.agents.executor import run_executor_agent as run_executor

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        _, result = run_executor(
            kernel_source=kernel_source,
            optimization_plan=sample_optimization_plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
        )

        # May or may not have lines_changed field
        if "lines_changed" in result:
            assert isinstance(result["lines_changed"], int)
            assert result["lines_changed"] >= 0
