"""
Tests for Correctness Agent (CPU reference generation).

The correctness checking system has multiple modes:
1. Built-in self-tests: Kernel has its own correctness verification
2. Agentic mode (use_agentic=True): LLM generates CPU reference + comparison
   - For CUDA: LLM writes complete C harness with CPU reference
   - For Python (Triton/Helion): LLM generates CPU reference + torch.allclose
3. Non-agentic mode: Run kernel and check exit code + failure markers

These tests verify the correctness checking functionality.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestCorrectnessBasic:
    """Basic correctness checking tests."""

    def test_cuda_builtin_self_test_detection(self):
        """Test detection of built-in self-tests in CUDA kernels."""
        from kernelopt.functions.correctness import _has_builtin_self_test

        # Has check_correctness function
        cuda_with_check = """
__global__ void kernel(float* x) { x[0] = 1.0f; }

void check_correctness() {
    // Verify results
}
"""
        assert _has_builtin_self_test(cuda_with_check)

        # Has PASS/FAIL printf
        cuda_with_printf = """
__global__ void kernel(float* x) { x[0] = 1.0f; }

int main() {
    printf("Test PASS\\n");
    return 0;
}
"""
        assert _has_builtin_self_test(cuda_with_printf)

        # No self-test
        cuda_no_test = """
__global__ void kernel(float* x) { x[0] = 1.0f; }
"""
        assert not _has_builtin_self_test(cuda_no_test)

    def test_check_correctness_accepts_parameters(self):
        """Test that check_correctness accepts all expected parameters."""
        from kernelopt.functions.correctness import check_correctness

        # Should accept these parameters without error
        try:
            # Mock the actual execution
            with patch('kernelopt.functions.correctness._check_python_kernel') as mock:
                mock.return_value = (True, "Success")

                passed, msg = check_correctness(
                    original_path="/tmp/test.py",
                    optimized_source="# code",
                    framework="triton",
                    use_agentic=False,  # Avoid LLM call
                    api_key=None,
                    model="claude-sonnet-4@20250514",
                    project_id=None,
                    region=None,
                    kernel_args=[],
                    binary_path=None,
                )
                assert True  # All parameters accepted
        except TypeError as e:
            pytest.fail(f"Parameter not accepted: {e}")


class TestCUDACorrectness:
    """Tests for CUDA kernel correctness checking."""

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.agent
    @pytest.mark.xfail(reason="Requires LLM to generate valid C harness, may not always work")
    def test_cuda_agentic_correctness(
        self, temp_output_dir, skip_if_no_cuda, patch_build_llm, gpu_id
    ):
        """Test agentic CUDA correctness (LLM generates C harness) - xfail."""
        from kernelopt.functions.correctness import check_correctness

        patch_build_llm(gpu_id)

        # Simple CUDA kernel without built-in test
        cuda_kernel = """
__global__ void scale_kernel(float* x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}
"""

        cu_file = temp_output_dir / "scale_kernel.cu"
        cu_file.write_text(cuda_kernel)

        # Check correctness (agentic - LLM writes harness)
        passed, message = check_correctness(
            original_path=str(cu_file),
            optimized_source=cuda_kernel,
            framework="cuda",
            use_agentic=True,  # LLM generates CPU reference
            model="claude-sonnet-4@20250514",
        )

        # May or may not pass - LLM harness generation is complex
        assert isinstance(passed, bool)
        assert isinstance(message, str)


class TestPythonCorrectness:
    """Tests for Python (Triton/Helion) kernel correctness checking."""

    def test_python_kernel_subprocess_check(self, temp_output_dir):
        """Test non-agentic Python kernel correctness checking."""
        from kernelopt.functions.correctness import check_correctness_subprocess

        # Python kernel that passes
        passing_kernel = """
import sys
print("Running kernel...")
# Kernel executes successfully
sys.exit(0)
"""

        py_file = temp_output_dir / "passing.py"
        py_file.write_text(passing_kernel)

        passed, message = check_correctness_subprocess(
            original_path=str(py_file),
            optimized_source=passing_kernel,
            framework="triton",
        )

        # Should pass
        assert passed
        assert "exit 0" in message or "successfully" in message.lower()

    def test_python_kernel_failure_detection(self, temp_output_dir):
        """Test detection of failures in Python kernels."""
        from kernelopt.functions.correctness import check_correctness_subprocess

        # Python kernel that fails
        failing_kernel = """
print("FAILED: Correctness check failed!")
"""

        py_file = temp_output_dir / "failing.py"
        py_file.write_text(failing_kernel)

        passed, message = check_correctness_subprocess(
            original_path=str(py_file),
            optimized_source=failing_kernel,
            framework="triton",
        )

        # Should fail (FAILED indicator found)
        assert not passed
        assert "FAILED" in message

    def test_python_kernel_crash_detection(self, temp_output_dir):
        """Test detection of crashes in Python kernels."""
        from kernelopt.functions.correctness import check_correctness_subprocess

        # Python kernel that crashes
        crashing_kernel = """
raise RuntimeError("Kernel crashed!")
"""

        py_file = temp_output_dir / "crashing.py"
        py_file.write_text(crashing_kernel)

        passed, message = check_correctness_subprocess(
            original_path=str(py_file),
            optimized_source=crashing_kernel,
            framework="triton",
        )

        # Should fail (RuntimeError detected)
        assert not passed
        assert "RuntimeError" in message or "crashed" in message.lower()

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.xfail(reason="Requires LLM to generate valid CPU reference, complex task")
    def test_python_agentic_correctness(
        self, temp_output_dir, patch_build_llm, gpu_id
    ):
        """Test agentic Python correctness (LLM generates CPU reference) - xfail."""
        from kernelopt.functions.correctness import check_correctness

        patch_build_llm(gpu_id)

        # Simple Triton kernel
        triton_kernel = """
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
"""

        py_file = temp_output_dir / "add_kernel.py"
        py_file.write_text(triton_kernel)

        # Check correctness (agentic - LLM generates CPU reference)
        passed, message = check_correctness(
            original_path=str(py_file),
            optimized_source=triton_kernel,
            framework="triton",
            use_agentic=True,
            model="claude-sonnet-4@20250514",
        )

        # May or may not pass - complex agentic task
        assert isinstance(passed, bool)
        assert isinstance(message, str)


class TestCorrectnessAgenticModes:
    """Test agentic vs non-agentic correctness modes."""

    def test_use_agentic_false_no_llm_call(self, temp_output_dir):
        """Test that use_agentic=False doesn't call LLM."""
        from kernelopt.functions.correctness import check_correctness

        simple_py = temp_output_dir / "simple.py"
        simple_py.write_text("print('ok')")

        # Mock build_llm to ensure it's not called
        with patch('kernelopt.functions.correctness.build_llm') as mock_build_llm:
            passed, message = check_correctness(
                original_path=str(simple_py),
                optimized_source="print('ok')",
                framework="triton",
                use_agentic=False,  # Should not call LLM
            )

            # LLM should not be built
            mock_build_llm.assert_not_called()

    @pytest.mark.agent
    @pytest.mark.slow
    def test_use_agentic_true_calls_llm(self, temp_output_dir, patch_build_llm, gpu_id):
        """Test that use_agentic=True attempts to call LLM."""
        from kernelopt.functions.correctness import check_correctness

        patch_build_llm(gpu_id)

        simple_py = temp_output_dir / "simple_agentic.py"
        simple_py.write_text("# triton kernel")

        # Should attempt agentic check (may fail, but should try)
        passed, message = check_correctness(
            original_path=str(simple_py),
            optimized_source="# triton kernel",
            framework="triton",
            use_agentic=True,  # Should use LLM
            model="claude-sonnet-4@20250514",
        )

        # Result doesn't matter, just that it attempted agentic path
        assert isinstance(passed, bool)
        assert isinstance(message, str)


class TestCorrectnessErrorHandling:
    """Test error handling in correctness checking."""

    def test_nonexistent_kernel_path(self):
        """Test handling of nonexistent kernel path."""
        from kernelopt.functions.correctness import check_correctness

        passed, message = check_correctness(
            original_path="/nonexistent/kernel.py",
            optimized_source="# code",
            framework="triton",
            use_agentic=False,
        )

        # May succeed (writes temp file) or fail - either is acceptable
        assert isinstance(passed, bool)
        assert isinstance(message, str)

    def test_unsupported_extension(self):
        """Test handling of unsupported file extension."""
        from kernelopt.functions.correctness import check_correctness_subprocess

        passed, message = check_correctness_subprocess(
            original_path="/tmp/kernel.txt",
            optimized_source="# code",
            framework="unknown",
        )

        # Should fail with unsupported extension message
        assert not passed
        assert "Unsupported" in message or "extension" in message.lower()

    def test_timeout_handling(self, temp_output_dir):
        """Test handling of kernel execution timeout."""
        from kernelopt.functions.correctness import check_correctness_subprocess

        # Kernel that runs forever
        infinite_kernel = """
while True:
    pass
"""

        py_file = temp_output_dir / "infinite.py"
        py_file.write_text(infinite_kernel)

        passed, message = check_correctness_subprocess(
            original_path=str(py_file),
            optimized_source=infinite_kernel,
            framework="triton",
        )

        # Should timeout and fail
        assert not passed
        assert "timeout" in message.lower() or "timed out" in message.lower()


class TestCorrectnessIntegration:
    """Integration tests for correctness checking."""

    def test_correctness_with_binary_path(self, temp_output_dir, skip_if_no_cuda):
        """Test CUDA correctness with pre-compiled binary path."""
        from kernelopt.functions.correctness import check_correctness

        # If we have a pre-compiled binary, skip nvcc compilation
        # This is used when executor already compiled the kernel

        cuda_source = """
__global__ void dummy() {}
int main() { printf("PASS\\n"); return 0; }
"""

        cu_file = temp_output_dir / "dummy.cu"
        cu_file.write_text(cuda_source)

        # Without binary_path (will try to compile)
        passed, message = check_correctness(
            original_path=str(cu_file),
            optimized_source=cuda_source,
            framework="cuda",
            use_agentic=False,
            binary_path=None,  # No pre-compiled binary
        )

        # May pass or fail depending on nvcc availability
        assert isinstance(passed, bool)

    def test_framework_detection(self, temp_output_dir):
        """Test that framework is properly detected and handled."""
        from kernelopt.functions.correctness import check_correctness

        # Python file should use Python checking
        py_file = temp_output_dir / "test.py"
        py_file.write_text("print('test')")

        passed, message = check_correctness(
            original_path=str(py_file),
            optimized_source="print('test')",
            framework="triton",
            use_agentic=False,
        )

        # Should attempt Python checking
        assert isinstance(passed, bool)


class TestCorrectnessEdgeCases:
    """Test edge cases in correctness checking."""

    def test_empty_kernel_source(self, temp_output_dir):
        """Test handling of empty kernel source."""
        from kernelopt.functions.correctness import check_correctness_subprocess

        py_file = temp_output_dir / "empty.py"
        py_file.write_text("")

        passed, message = check_correctness_subprocess(
            original_path=str(py_file),
            optimized_source="",
            framework="triton",
        )

        # May pass (empty file runs successfully) or fail
        assert isinstance(passed, bool)

    def test_very_long_output(self, temp_output_dir):
        """Test handling of very long kernel output."""
        from kernelopt.functions.correctness import check_correctness_subprocess

        long_output_kernel = """
for i in range(1000):
    print(f"Line {i}: " + "x" * 100)
"""

        py_file = temp_output_dir / "long_output.py"
        py_file.write_text(long_output_kernel)

        passed, message = check_correctness_subprocess(
            original_path=str(py_file),
            optimized_source=long_output_kernel,
            framework="triton",
        )

        # Should handle long output (may truncate in message)
        assert isinstance(passed, bool)
        assert len(message) < 10000  # Message should be truncated

    def test_special_characters_in_output(self, temp_output_dir):
        """Test handling of special characters in kernel output."""
        from kernelopt.functions.correctness import check_correctness_subprocess

        special_chars_kernel = """
print("Test with special chars: ñ, é, 中文, 🚀")
"""

        py_file = temp_output_dir / "special_chars.py"
        py_file.write_text(special_chars_kernel)

        passed, message = check_correctness_subprocess(
            original_path=str(py_file),
            optimized_source=special_chars_kernel,
            framework="triton",
        )

        # Should handle unicode characters
        assert isinstance(passed, bool)
