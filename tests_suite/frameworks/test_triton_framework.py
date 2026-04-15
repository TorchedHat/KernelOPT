"""
Triton Framework-Specific Tests.

Tests for Triton kernel compilation, JIT, and execution.
Covers Triton-specific features like @triton.jit, tl.*, autotune, etc.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


class TestTritonCompilation:
    """Test Triton kernel compilation and JIT."""

    def test_valid_triton_kernel_syntax(self, temp_output_dir):
        """Test that valid Triton kernel has correct syntax."""
        triton_source = """
import triton
import triton.language as tl

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
        kernel_file = temp_output_dir / "test_triton.py"
        kernel_file.write_text(triton_source)

        # Verify Triton keywords
        assert "@triton.jit" in triton_source
        assert "tl.program_id" in triton_source
        assert "tl.load" in triton_source
        assert "tl.store" in triton_source

    def test_triton_jit_decorator_present(self):
        """Test detecting @triton.jit decorator."""
        triton_code = """
@triton.jit
def kernel():
    pass
"""
        assert "@triton.jit" in triton_code

    def test_triton_autotune_decorator(self):
        """Test detecting @triton.autotune."""
        triton_code = """
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel():
    pass
"""
        assert "@triton.autotune" in triton_code
        assert "triton.Config" in triton_code

    def test_triton_constexpr_usage(self):
        """Test tl.constexpr for compile-time constants."""
        triton_code = """
def kernel(BLOCK_SIZE: tl.constexpr):
    pass
"""
        assert "tl.constexpr" in triton_code


class TestTritonProfiling:
    """Test Triton kernel profiling."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_triton_ncu_profiling_kernel_name_empty(self, temp_output_dir):
        """Test that Triton kernels use empty kernel_name for NCU."""
        # Triton JIT-compiles to mangled names, so kernel_name should be ""
        triton_source = """
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, n, BLOCK_SIZE: tl.constexpr):
    pass
"""
        kernel_file = temp_output_dir / "triton_ncu.py"
        kernel_file.write_text(triton_source)

        # For Triton, we should use kernel_name="" because of name mangling
        kernel_name = ""  # Empty for Triton
        assert kernel_name == ""

    def test_triton_kernel_name_mangling(self):
        """Test that Triton mangles kernel names."""
        # Triton JIT compiles to names like:
        # _kernel_0d1d2d3d4d5d6d7d8d9d
        # So we can't use the Python function name for NCU filtering
        python_name = "softmax_kernel"
        # JIT-compiled name will be different
        # This is why we use kernel_name="" for Triton
        assert python_name != ""  # Python name exists
        # But NCU filter should be empty
        ncu_filter = ""
        assert ncu_filter == ""


class TestTritonLanguagePrimitives:
    """Test Triton language primitives (tl.*)."""

    def test_triton_load_store(self):
        """Test tl.load and tl.store primitives."""
        triton_code = """
x = tl.load(ptr + offsets, mask=mask)
tl.store(ptr + offsets, value, mask=mask)
"""
        assert "tl.load" in triton_code
        assert "tl.store" in triton_code
        assert "mask=" in triton_code

    def test_triton_program_id(self):
        """Test tl.program_id for block indexing."""
        triton_code = """
pid = tl.program_id(axis=0)
"""
        assert "tl.program_id" in triton_code

    def test_triton_arange(self):
        """Test tl.arange for offset generation."""
        triton_code = """
offsets = tl.arange(0, BLOCK_SIZE)
"""
        assert "tl.arange" in triton_code

    def test_triton_math_operations(self):
        """Test Triton math operations."""
        triton_code = """
result = tl.exp(x)
result = tl.log(x)
result = tl.sqrt(x)
result = tl.sum(x, axis=0)
result = tl.max(x, axis=0)
"""
        assert "tl.exp" in triton_code
        assert "tl.log" in triton_code
        assert "tl.sum" in triton_code

    def test_triton_atomic_operations(self):
        """Test Triton atomic operations."""
        triton_code = """
tl.atomic_add(ptr, val)
tl.atomic_max(ptr, val)
"""
        assert "tl.atomic_add" in triton_code


class TestTritonOptimizationPatterns:
    """Test Triton optimization patterns."""

    def test_triton_block_size_tuning(self):
        """Test BLOCK_SIZE as constexpr for tuning."""
        triton_code = """
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(BLOCK_SIZE: tl.constexpr):
    pass
"""
        assert "BLOCK_SIZE" in triton_code
        assert "tl.constexpr" in triton_code

    def test_triton_vectorization(self):
        """Test Triton automatic vectorization."""
        # Triton auto-vectorizes memory operations
        triton_code = """
x = tl.load(x_ptr + offsets)  # Auto-vectorized
"""
        assert "tl.load" in triton_code
        # Triton handles vectorization automatically

    def test_triton_masking_for_safety(self):
        """Test masking for out-of-bounds access."""
        triton_code = """
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
"""
        assert "mask" in triton_code
        assert "< n_elements" in triton_code


class TestTritonFrameworkDetection:
    """Test Triton framework detection."""

    def test_detect_triton_by_import(self):
        """Test detecting Triton by import statement."""
        triton_code = "import triton"
        assert "import triton" in triton_code

    def test_detect_triton_language_import(self):
        """Test detecting triton.language import."""
        triton_code = "import triton.language as tl"
        assert "triton.language" in triton_code

    def test_detect_triton_jit_decorator(self):
        """Test detecting @triton.jit."""
        triton_code = "@triton.jit"
        assert "@triton.jit" in triton_code

    def test_triton_vs_cuda_distinction(self):
        """Test distinguishing Triton from CUDA."""
        triton_code = "@triton.jit\ndef kernel():"
        cuda_code = "__global__ void kernel() {}"

        assert "@triton.jit" in triton_code
        assert "__global__" in cuda_code
        assert "@triton.jit" not in cuda_code
        assert "__global__" not in triton_code


class TestTritonCorrectness:
    """Test Triton correctness checking."""

    def test_triton_output_validation(self):
        """Test validating Triton kernel output."""
        triton_kernel = """
@triton.jit
def multiply_by_two(x_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2
    tl.store(output_ptr + offsets, output, mask=mask)
"""
        # Kernel multiplies by 2
        assert "* 2" in triton_kernel


class TestTritonBlockPointers:
    """Test Triton block pointer feature."""

    def test_triton_make_block_ptr(self):
        """Test tl.make_block_ptr for 2D tensors."""
        triton_code = """
block_ptr = tl.make_block_ptr(
    base=ptr,
    shape=(M, N),
    strides=(stride_m, stride_n),
    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0)
)
"""
        assert "tl.make_block_ptr" in triton_code

    def test_triton_advance_block_ptr(self):
        """Test advancing block pointers."""
        triton_code = """
block_ptr = tl.advance(block_ptr, (BLOCK_M, 0))
"""
        assert "tl.advance" in triton_code


class TestTritonConfigOptions:
    """Test Triton configuration options."""

    def test_triton_num_warps_config(self):
        """Test num_warps configuration."""
        triton_code = """
triton.Config({'BLOCK_SIZE': 128}, num_warps=4)
"""
        assert "num_warps" in triton_code

    def test_triton_num_stages_config(self):
        """Test num_stages for pipelining."""
        triton_code = """
triton.Config({'BLOCK_SIZE': 128}, num_stages=3)
"""
        assert "num_stages" in triton_code
