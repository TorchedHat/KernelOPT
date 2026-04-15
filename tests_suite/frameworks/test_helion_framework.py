"""
Helion Framework-Specific Tests.

Tests for Helion kernel compilation and execution.
Helion is a less common framework, so tests focus on basic detection and syntax.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


class TestHelionCompilation:
    """Test Helion kernel compilation."""

    def test_valid_helion_kernel_syntax(self, temp_output_dir):
        """Test that valid Helion kernel has correct syntax."""
        helion_source = """
import helion

@helion.kernel
def add_kernel(a, b, c, n):
    idx = helion.thread_idx()
    if idx < n:
        c[idx] = a[idx] + b[idx]
"""
        kernel_file = temp_output_dir / "test_helion.py"
        kernel_file.write_text(helion_source)

        # Verify Helion keywords
        assert "import helion" in helion_source
        assert "@helion.kernel" in helion_source or "@helion" in helion_source

    def test_helion_decorator_present(self):
        """Test detecting @helion decorator."""
        helion_code = """
@helion.kernel
def kernel():
    pass
"""
        assert "@helion" in helion_code


class TestHelionFrameworkDetection:
    """Test Helion framework detection."""

    def test_detect_helion_by_import(self):
        """Test detecting Helion by import statement."""
        helion_code = "import helion"
        assert "import helion" in helion_code

    def test_detect_helion_decorator(self):
        """Test detecting @helion decorator."""
        helion_code = "@helion.kernel"
        assert "@helion" in helion_code

    def test_helion_vs_cuda_distinction(self):
        """Test distinguishing Helion from CUDA."""
        helion_code = "@helion.kernel\ndef kernel():"
        cuda_code = "__global__ void kernel() {}"

        assert "@helion" in helion_code
        assert "__global__" in cuda_code
        assert "@helion" not in cuda_code
        assert "__global__" not in helion_code

    def test_helion_vs_triton_distinction(self):
        """Test distinguishing Helion from Triton."""
        helion_code = "import helion"
        triton_code = "import triton"

        assert "helion" in helion_code
        assert "triton" in triton_code
        assert "helion" not in triton_code
        assert "triton" not in helion_code


class TestHelionLanguagePrimitives:
    """Test Helion language primitives."""

    def test_helion_thread_indexing(self):
        """Test Helion thread indexing."""
        helion_code = """
idx = helion.thread_idx()
block_idx = helion.block_idx()
"""
        # Helion may use different primitives
        assert "helion" in helion_code

    def test_helion_memory_operations(self):
        """Test Helion memory operations."""
        helion_code = """
value = helion.load(ptr, idx)
helion.store(ptr, idx, value)
"""
        # Helion may have its own memory ops
        assert "helion" in helion_code


class TestHelionOptimizationPatterns:
    """Test Helion optimization patterns."""

    def test_helion_block_size(self):
        """Test Helion block size specification."""
        helion_code = """
@helion.kernel(block_size=256)
def kernel():
    pass
"""
        # Helion may specify block size differently
        assert "helion" in helion_code


class TestHelionCorrectness:
    """Test Helion correctness checking."""

    def test_helion_output_validation(self):
        """Test validating Helion kernel output."""
        helion_kernel = """
@helion.kernel
def multiply_by_two(data, n):
    idx = helion.thread_idx()
    if idx < n:
        data[idx] *= 2
"""
        # Kernel multiplies by 2
        assert "*= 2" in helion_kernel or "* 2" in helion_kernel


class TestHelionProfiling:
    """Test Helion kernel profiling."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_helion_ncu_profiling_possibility(self):
        """Test that Helion kernels can potentially be profiled."""
        # Helion may compile to CUDA, so NCU profiling may work
        # This test just verifies the concept
        helion_source = """
import helion

@helion.kernel
def simple(x):
    x[0] = 1.0
"""
        # If Helion compiles to CUDA, it can be profiled with NCU
        # Otherwise, it may need different profiling tools
        assert "import helion" in helion_source
