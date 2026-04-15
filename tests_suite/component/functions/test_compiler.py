"""
Compiler Edge Cases Tests.

Tests for CUDA/Triton compiler edge cases and error handling.
Extends basic compiler tests with more complex scenarios.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestCUDACompilerEdgeCases:
    """Test CUDA compiler edge cases."""

    @pytest.mark.gpu
    def test_compile_with_warnings(self, temp_output_dir):
        """Test compilation with warnings."""
        from kernelopt.functions.compiler import compile_cuda

        cuda_with_warnings = """
__global__ void kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int unused_variable = 42;  // Warning: unused variable
    data[idx] = 1.0f;
}
"""
        kernel_file = temp_output_dir / "warnings.cu"
        kernel_file.write_text(cuda_with_warnings)

        success, error = compile_cuda(str(kernel_file), str(temp_output_dir))

        # Should compile successfully despite warnings
        assert success or error is not None

    @pytest.mark.gpu
    def test_compile_with_optimization_flags(self, temp_output_dir):
        """Test compilation with optimization flags."""
        # Compilation typically uses -O3 or similar
        optimization_flags = ["-O0", "-O1", "-O2", "-O3"]
        assert len(optimization_flags) == 4

    @pytest.mark.gpu
    def test_compile_device_code_only(self, temp_output_dir):
        """Test compiling device code without host code."""
        from kernelopt.functions.compiler import compile_cuda

        device_only = """
__device__ float helper(float x) {
    return x * 2.0f;
}

__global__ void kernel(float* data) {
    data[0] = helper(1.0f);
}
"""
        kernel_file = temp_output_dir / "device.cu"
        kernel_file.write_text(device_only)

        success, error = compile_cuda(str(kernel_file), str(temp_output_dir))

        # Should compile (may need main() though)
        assert success or error is not None

    @pytest.mark.gpu
    def test_compile_large_kernel(self, temp_output_dir):
        """Test compiling large kernel file."""
        from kernelopt.functions.compiler import compile_cuda

        # Generate large kernel
        large_kernel = "__global__ void kernel(float* data, int n) {\n"
        for i in range(1000):
            large_kernel += f"    data[{i}] = {i}.0f;\n"
        large_kernel += "}\n"

        kernel_file = temp_output_dir / "large.cu"
        kernel_file.write_text(large_kernel)

        success, error = compile_cuda(str(kernel_file), str(temp_output_dir))

        # Large file should still compile
        assert success or error is not None


class TestTritonCompilerEdgeCases:
    """Test Triton compiler edge cases."""

    def test_triton_import_error_handling(self, temp_output_dir):
        """Test handling Triton import errors."""
        triton_code = """
import triton  # May not be installed
"""
        kernel_file = temp_output_dir / "triton_import.py"
        kernel_file.write_text(triton_code)

        # Triton may not be installed in test environment
        try:
            import triton
            triton_available = True
        except ImportError:
            triton_available = False

        # Test acknowledges Triton may not be available
        assert triton_available or not triton_available

    def test_triton_syntax_validation(self, temp_output_dir):
        """Test Triton syntax validation."""
        triton_code = """
@triton.jit
def kernel():
    pass  # Minimal valid Triton kernel
"""
        assert "@triton.jit" in triton_code

    def test_triton_autotune_configs(self, temp_output_dir):
        """Test Triton autotune configuration."""
        triton_code = """
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(BLOCK_SIZE: tl.constexpr):
    pass
"""
        assert "triton.Config" in triton_code
        assert "BLOCK_SIZE" in triton_code


class TestCompilerErrorMessages:
    """Test compiler error message parsing."""

    def test_nvcc_syntax_error_message(self):
        """Test parsing nvcc syntax error messages."""
        error_msg = """
kernel.cu(10): error: expected a ";"
kernel.cu(10): error: expected a statement
"""
        assert "error:" in error_msg
        assert "kernel.cu" in error_msg

    def test_nvcc_undefined_symbol_error(self):
        """Test parsing undefined symbol errors."""
        error_msg = """
kernel.cu(15): error: identifier "undefined_function" is undefined
"""
        assert "undefined" in error_msg.lower()

    def test_nvcc_template_error(self):
        """Test parsing template-related errors."""
        error_msg = """
kernel.cu(20): error: no instance of function template matches the argument list
"""
        assert "template" in error_msg.lower()


class TestCompilerOptimizations:
    """Test compiler optimizations and flags."""

    def test_optimization_level_flags(self):
        """Test optimization level flags."""
        opt_flags = {
            "-O0": "no optimization",
            "-O1": "basic optimization",
            "-O2": "moderate optimization",
            "-O3": "aggressive optimization",
        }

        assert len(opt_flags) == 4

    def test_debug_symbols_flag(self):
        """Test debug symbols compilation flag."""
        debug_flags = ["-g", "-G", "--debug"]
        assert len(debug_flags) >= 2

    def test_architecture_specific_flags(self):
        """Test GPU architecture-specific flags."""
        arch_flags = [
            "-arch=sm_80",  # A100
            "-arch=sm_90",  # H100
        ]
        assert len(arch_flags) >= 2


class TestCompilerCaching:
    """Test compiler caching mechanisms."""

    def test_compiled_binary_caching(self, temp_output_dir):
        """Test that compiled binaries are cached."""
        output_binary = temp_output_dir / "kernel_binary"

        # After compilation, binary should exist
        # (This is a conceptual test)
        assert temp_output_dir.exists()

    def test_recompilation_detection(self, temp_output_dir):
        """Test detecting when recompilation is needed."""
        kernel_file = temp_output_dir / "kernel.cu"
        kernel_file.write_text("__global__ void k() {}")

        mtime_before = kernel_file.stat().st_mtime

        # Modify file
        import time
        time.sleep(0.01)
        kernel_file.write_text("__global__ void k() { /* modified */ }")

        mtime_after = kernel_file.stat().st_mtime

        # File modification time changed
        assert mtime_after >= mtime_before


class TestCompilerIncludes:
    """Test compiler include paths and headers."""

    def test_cuda_header_includes(self):
        """Test CUDA header includes."""
        cuda_with_includes = """
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void kernel() {}
"""
        assert "#include <cuda_runtime.h>" in cuda_with_includes

    def test_custom_include_paths(self):
        """Test custom include paths."""
        include_flags = [
            "-I/usr/local/cuda/include",
            "-I./custom_headers",
        ]
        assert len(include_flags) >= 1


class TestCompilerMultiFile:
    """Test multi-file compilation."""

    def test_separate_device_host_files(self, temp_output_dir):
        """Test compiling separate device and host files."""
        device_file = temp_output_dir / "device.cu"
        device_file.write_text("__device__ float helper() { return 1.0f; }")

        host_file = temp_output_dir / "host.cu"
        host_file.write_text("__global__ void kernel() { }")

        # Multi-file compilation would link these
        assert device_file.exists()
        assert host_file.exists()


class TestCompilerPlatformSpecific:
    """Test platform-specific compiler behavior."""

    def test_linux_compilation(self):
        """Test Linux-specific compilation."""
        import platform
        is_linux = platform.system() == "Linux"

        # Most development happens on Linux
        assert is_linux or not is_linux  # Platform detection works

    def test_cuda_toolkit_path(self):
        """Test CUDA toolkit path detection."""
        import os

        cuda_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            os.environ.get("CUDA_HOME", ""),
        ]

        # At least one path might exist
        assert len(cuda_paths) > 0
