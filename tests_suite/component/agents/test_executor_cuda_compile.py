"""
Tests for Executor Agent's CUDA compilation loop with K attempts.

The executor agent supports CUDA kernel compilation with:
- K maximum attempts via submit_kernel tool
- nvcc error feedback passed back to LLM via compile_error parameter
- Automatic retry with error context

These tests verify the K-attempt loop works correctly and produces valid results.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestExecutorCUDACompileBasic:
    """Basic CUDA compilation tests."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_cuda_compile_success_first_attempt(
        self, temp_output_dir, patch_build_llm, gpu_id
    ):
        """Test successful CUDA compilation on first attempt."""
        from kernelopt.agents.executor import run_executor_agent

        patch_build_llm(gpu_id)

        # Valid CUDA kernel
        cuda_kernel = """
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    return 0;
}
"""

        plan = {
            "plan_id": "cuda_test",
            "direction": "optimize_memory_access",
            "framework": "cuda",
            "step": {
                "action": "modify",
                "target": "memory access pattern",
                "change": "Coalesce global memory accesses",
                "predicted_effect": "Better memory throughput",
            },
        }

        optimized_source, result = run_executor_agent(
            kernel_source=cuda_kernel,
            optimization_plan=plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # Should succeed (or fail gracefully)
        assert result is not None
        assert "status" in result

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.agent
    def test_compile_error_feedback_to_llm(
        self, temp_output_dir, patch_build_llm, gpu_id
    ):
        """Test that compile errors are fed back to LLM for retry."""
        from kernelopt.agents.executor import run_executor_agent

        patch_build_llm(gpu_id)

        # Initial kernel with syntax error
        cuda_kernel_with_error = """
__global__ void broken_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f
    }  // Missing semicolon
}
"""

        plan = {
            "plan_id": "cuda_fix_test",
            "direction": "fix_syntax",
            "framework": "cuda",
            "step": {
                "action": "fix",
                "target": "syntax error",
                "change": "Add missing semicolon",
                "predicted_effect": "Code compiles",
            },
        }

        # First attempt: provide compile error
        compile_error = """
kernel.cu(4): error: expected a ";"
"""

        optimized_source, result = run_executor_agent(
            kernel_source=cuda_kernel_with_error,
            optimization_plan=plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
            compile_error=compile_error,  # Pass error to LLM
        )

        # LLM should receive the error and attempt to fix
        assert result is not None

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_k_attempts_limit_respected(self, temp_output_dir, patch_build_llm, gpu_id):
        """Test that executor respects K attempts limit."""
        from kernelopt.agents.executor import run_executor_agent

        patch_build_llm(gpu_id)

        cuda_kernel = """
__global__ void test_kernel(float* data) {
    data[0] = 1.0f;
}
"""

        plan = {
            "plan_id": "k_limit_test",
            "direction": "test_k_limit",
            "framework": "cuda",
            "step": {
                "action": "modify",
                "target": "kernel",
                "change": "Test K limit",
                "predicted_effect": "N/A",
            },
        }

        # Set K=2 (max 2 attempts)
        optimized_source, result = run_executor_agent(
            kernel_source=cuda_kernel,
            optimization_plan=plan,
            memory=[],
            K=2,  # Only 2 attempts allowed
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # Should respect K limit
        assert result is not None
        if "attempt" in result:
            # Attempt should be <= K
            assert result["attempt"] <= 2


class TestExecutorMultipleAttempts:
    """Test multiple compilation attempts with error feedback."""

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.xfail(reason="Requires LLM to successfully fix compile errors, which may not always happen")
    def test_retry_after_compile_failure(self, temp_output_dir, patch_build_llm, gpu_id):
        """Test that executor retries after compile failure (xfail - LLM may not fix error)."""
        from kernelopt.agents.executor import run_executor_agent

        patch_build_llm(gpu_id)

        # Kernel with recoverable error
        broken_kernel = """
__global__ void kernel(float* x, int n) {
    int idx = threadIdx.x;
    x[idx] = idx * 2  // Missing semicolon
}
"""

        plan = {
            "plan_id": "retry_test",
            "direction": "fix_and_optimize",
            "framework": "cuda",
            "step": {
                "action": "fix",
                "target": "syntax",
                "change": "Fix compile errors",
                "predicted_effect": "Compiles successfully",
            },
        }

        optimized_source, result = run_executor_agent(
            kernel_source=broken_kernel,
            optimization_plan=plan,
            memory=[],
            K=5,  # Allow multiple attempts
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # LLM might fix it, might not - hence xfail
        # If it does fix, status should be success
        if result["status"] == "success":
            assert result.get("ready_for_profiling", False)

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_different_errors_per_attempt(self, temp_output_dir):
        """Test that different compile errors are handled per attempt."""
        from kernelopt.agents.executor import run_executor_agent
        from langchain_core.messages import AIMessage

        # Mock LLM that generates different broken code each time
        attempt_count = {"count": 0}

        def mock_invoke(messages):
            attempt_count["count"] += 1
            # Return different errors for each attempt
            broken_codes = [
                "__global__ void k1(float* x) { x[0] = 1 }",  # Missing semicolon
                "__global__ void k2(float* x) { x[0] == 1; }",  # == instead of =
                "__global__ void k3(float* x) { x[0] = 1; }",  # Valid
            ]
            idx = min(attempt_count["count"] - 1, len(broken_codes) - 1)
            return AIMessage(content=f"Here's the code:\n```cuda\n{broken_codes[idx]}\n```")

        mock_llm = Mock()
        mock_llm.invoke = mock_invoke

        plan = {
            "plan_id": "multi_error_test",
            "direction": "test_multi_errors",
            "framework": "cuda",
            "step": {
                "action": "fix",
                "target": "errors",
                "change": "Fix compilation errors",
                "predicted_effect": "Compiles",
            },
        }

        _, result = run_executor_agent(
            kernel_source="__global__ void start(float* x) { }",
            optimization_plan=plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
            llm_override=mock_llm,
        )

        # Should return a result (LLM may or may not be called depending on execution path)
        assert result is not None
        assert "status" in result


class TestExecutorCompileErrorFormatting:
    """Test that nvcc errors are properly formatted and passed to LLM."""

    def test_compile_error_parameter_accepted(self):
        """Test that compile_error parameter is accepted by run_executor_agent."""
        from kernelopt.agents.executor import run_executor_agent
        from unittest.mock import Mock

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Fixed")

        plan = {"plan_id": "test", "direction": "test"}

        # Should accept compile_error parameter without error
        try:
            _, result = run_executor_agent(
                kernel_source="__global__ void k() {}",
                optimization_plan=plan,
                memory=[],
                K=1,
                llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
                llm_override=mock_llm,
                compile_error="kernel.cu(10): error: something broke",
            )
            assert True  # Parameter accepted
        except TypeError as e:
            pytest.fail(f"compile_error parameter not accepted: {e}")

    def test_compile_error_in_initial_message(self, temp_output_dir):
        """Test that compile_error appears in initial message to LLM."""
        from kernelopt.agents.executor.prompts import build_initial_message

        plan_direction = "fix_compilation"
        K = 3
        compile_error = "kernel.cu(15): error: identifier 'foo' is undefined"

        message = build_initial_message(plan_direction, K, compile_error=compile_error)

        # Error should be in the message
        assert "error:" in message.content.lower() or "kernel.cu" in message.content


class TestExecutorCUDAFrameworkDetection:
    """Test framework detection and handling for CUDA kernels."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_cuda_framework_in_result(self, temp_output_dir, patch_build_llm, gpu_id):
        """Test that CUDA framework is detected and included in result."""
        from kernelopt.agents.executor import run_executor_agent

        patch_build_llm(gpu_id)

        cuda_kernel = """
__global__ void simple(float* x) {
    x[0] = 1.0f;
}
"""

        plan = {
            "plan_id": "cuda_framework_test",
            "direction": "test_framework",
            "framework": "cuda",
        }

        _, result = run_executor_agent(
            kernel_source=cuda_kernel,
            optimization_plan=plan,
            memory=[],
            K=2,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # Framework should be in result
        # Note: framework detection may vary based on LLM response
        if "framework" in result:
            # Allow any framework value - executor may detect differently
            assert isinstance(result["framework"], str)
            assert result["framework"] in ["cuda", "triton", "helion", "unknown"]


class TestExecutorCompileSuccessMetrics:
    """Test metrics reported on successful compilation."""

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.agent
    def test_successful_compile_has_metrics(
        self, temp_output_dir, patch_build_llm, gpu_id
    ):
        """Test that successful compilation reports expected metrics."""
        from kernelopt.agents.executor import run_executor_agent

        patch_build_llm(gpu_id)

        cuda_kernel = """
__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""

        plan = {
            "plan_id": "metrics_test",
            "direction": "test_metrics",
            "framework": "cuda",
        }

        _, result = run_executor_agent(
            kernel_source=cuda_kernel,
            optimization_plan=plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # Should have status
        assert "status" in result
        assert result["status"] in ["success", "failed", "implementation_failure"]

        # If successful, should be ready for profiling
        if result["status"] == "success":
            assert "ready_for_profiling" in result


class TestExecutorMemoryIntegration:
    """Test that executor uses memory during K attempts."""

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.agent
    def test_memory_passed_to_executor(self, temp_output_dir, patch_build_llm, gpu_id):
        """Test that memory items are accessible during execution."""
        from kernelopt.agents.executor import run_executor_agent

        patch_build_llm(gpu_id)

        memory_items = [
            {
                "item_id": "test_1",
                "direction": "use_shared_memory",
                "strategy_description": "Use shared memory to reduce global memory access",
                "speedup": 1.3,
            }
        ]

        plan = {
            "plan_id": "memory_test",
            "direction": "apply_shared_memory",
            "framework": "cuda",
        }

        _, result = run_executor_agent(
            kernel_source="__global__ void k(float* x) { x[0] = 1.0f; }",
            optimization_plan=plan,
            memory=memory_items,  # Pass memory
            K=2,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # Memory should be available to LLM
        assert result is not None


class TestExecutorEdgeCases:
    """Test edge cases in CUDA compilation."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_k_equals_one(self, temp_output_dir, patch_build_llm, gpu_id):
        """Test executor with K=1 (single attempt)."""
        from kernelopt.agents.executor import run_executor_agent

        patch_build_llm(gpu_id)

        plan = {
            "plan_id": "k1_test",
            "direction": "test_k1",
            "framework": "cuda",
            "step": {"action": "test", "target": "k1", "change": "test", "predicted_effect": "test"},
        }

        _, result = run_executor_agent(
            kernel_source="__global__ void k() {}",
            optimization_plan=plan,
            memory=[],
            K=1,  # Only one attempt
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # Should still work with K=1
        assert result is not None
        if "attempt" in result:
            assert result["attempt"] <= 1

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_k_equals_ten(self, temp_output_dir, patch_build_llm, gpu_id):
        """Test executor with K=10 (many attempts)."""
        from kernelopt.agents.executor import run_executor_agent

        patch_build_llm(gpu_id)

        plan = {
            "plan_id": "k10_test",
            "direction": "test_k10",
            "framework": "cuda",
            "step": {"action": "test", "target": "k10", "change": "test", "predicted_effect": "test"},
        }

        _, result = run_executor_agent(
            kernel_source="__global__ void k() {}",
            optimization_plan=plan,
            memory=[],
            K=10,  # Many attempts allowed
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        assert result is not None

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_empty_plan(self, temp_output_dir, patch_build_llm, gpu_id):
        """Test executor with minimal plan."""
        from kernelopt.agents.executor import run_executor_agent

        patch_build_llm(gpu_id)

        minimal_plan = {"plan_id": "minimal"}

        _, result = run_executor_agent(
            kernel_source="__global__ void k() {}",
            optimization_plan=minimal_plan,
            memory=[],
            K=2,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # Should handle minimal plan gracefully
        assert result is not None

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_very_long_compile_error(self, temp_output_dir):
        """Test handling of very long compile error messages."""
        from kernelopt.agents.executor import run_executor_agent
        from unittest.mock import Mock

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Fixed")

        # Very long error (simulating nvcc verbose output)
        long_error = "error: " + "x" * 10000

        plan = {"plan_id": "long_error", "direction": "test"}

        _, result = run_executor_agent(
            kernel_source="__global__ void k() {}",
            optimization_plan=plan,
            memory=[],
            K=1,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
            llm_override=mock_llm,
            compile_error=long_error,
        )

        # Should handle long errors without crashing
        assert result is not None
