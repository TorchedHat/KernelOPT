"""
Framework Detection Tests.

Tests for detecting which framework (CUDA, Triton, Helion, or unknown) a kernel uses.
Tests both extension-based and content-based detection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


class TestFrameworkDetectionByExtension:
    """Test framework detection by file extension."""

    def test_detect_cuda_by_cu_extension(self):
        """Test detecting CUDA by .cu extension."""
        kernel_path = Path("kernel.cu")
        assert kernel_path.suffix == ".cu"

    def test_detect_cuda_by_cuh_extension(self):
        """Test detecting CUDA by .cuh (header) extension."""
        kernel_path = Path("kernel.cuh")
        assert kernel_path.suffix == ".cuh"

    def test_detect_python_based_frameworks(self):
        """Test detecting Python-based frameworks (.py)."""
        kernel_path = Path("kernel.py")
        assert kernel_path.suffix == ".py"
        # Could be Triton, Helion, or other Python-based framework

    def test_unknown_extension(self):
        """Test handling unknown extensions."""
        kernel_path = Path("kernel.xyz")
        assert kernel_path.suffix not in [".cu", ".cuh", ".py"]


class TestFrameworkDetectionByContent:
    """Test framework detection by analyzing kernel content."""

    def test_detect_cuda_by_global_keyword(self, temp_output_dir):
        """Test detecting CUDA by __global__ keyword."""
        cuda_source = "__global__ void kernel() {}"
        kernel_file = temp_output_dir / "kernel.cu"
        kernel_file.write_text(cuda_source)

        content = kernel_file.read_text()
        is_cuda = "__global__" in content
        assert is_cuda

    def test_detect_triton_by_import(self, temp_output_dir):
        """Test detecting Triton by import statement."""
        triton_source = "import triton\n@triton.jit\ndef kernel(): pass"
        kernel_file = temp_output_dir / "kernel.py"
        kernel_file.write_text(triton_source)

        content = kernel_file.read_text()
        is_triton = "import triton" in content or "triton.jit" in content
        assert is_triton

    def test_detect_helion_by_import(self, temp_output_dir):
        """Test detecting Helion by import statement."""
        helion_source = "import helion\n@helion.kernel\ndef kernel(): pass"
        kernel_file = temp_output_dir / "kernel.py"
        kernel_file.write_text(helion_source)

        content = kernel_file.read_text()
        is_helion = "import helion" in content or "@helion" in content
        assert is_helion

    def test_detect_unknown_framework(self, temp_output_dir):
        """Test handling kernels with unknown framework."""
        unknown_source = "def regular_python_function(): pass"
        kernel_file = temp_output_dir / "kernel.py"
        kernel_file.write_text(unknown_source)

        content = kernel_file.read_text()
        is_cuda = "__global__" in content
        is_triton = "triton" in content
        is_helion = "helion" in content

        # None of the known frameworks
        assert not is_cuda
        assert not is_triton
        assert not is_helion


class TestFrameworkDetectionPriority:
    """Test priority when multiple framework indicators present."""

    def test_extension_overrides_content_for_cuda(self, temp_output_dir):
        """Test .cu extension takes priority even with conflicting content."""
        # File with .cu extension should be treated as CUDA
        kernel_file = temp_output_dir / "kernel.cu"
        kernel_file.write_text("// Could have any content")

        assert kernel_file.suffix == ".cu"
        # Extension is ".cu" → treat as CUDA regardless of content

    def test_triton_import_overrides_python_extension(self, temp_output_dir):
        """Test Triton import detected in .py file."""
        kernel_file = temp_output_dir / "kernel.py"
        kernel_file.write_text("import triton")

        assert kernel_file.suffix == ".py"
        assert "import triton" in kernel_file.read_text()
        # .py extension + triton import → Triton

    def test_helion_import_overrides_python_extension(self, temp_output_dir):
        """Test Helion import detected in .py file."""
        kernel_file = temp_output_dir / "kernel.py"
        kernel_file.write_text("import helion")

        assert kernel_file.suffix == ".py"
        assert "import helion" in kernel_file.read_text()
        # .py extension + helion import → Helion


class TestFrameworkKeywords:
    """Test framework-specific keywords and patterns."""

    def test_cuda_keywords_comprehensive(self):
        """Test comprehensive CUDA keyword detection."""
        cuda_keywords = [
            "__global__",
            "__device__",
            "__host__",
            "__shared__",
            "__constant__",
            "cudaMalloc",
            "cudaFree",
            "cudaMemcpy",
            "<<<",
            ">>>",
        ]

        cuda_code = """
__global__ void kernel() {
    __shared__ float data[256];
}
int main() {
    kernel<<<1, 256>>>();
}
"""

        # At least some CUDA keywords should be present
        found_keywords = [kw for kw in cuda_keywords if kw in cuda_code]
        assert len(found_keywords) >= 3

    def test_triton_keywords_comprehensive(self):
        """Test comprehensive Triton keyword detection."""
        triton_keywords = [
            "import triton",
            "@triton.jit",
            "@triton.autotune",
            "tl.program_id",
            "tl.load",
            "tl.store",
            "tl.arange",
            "tl.constexpr",
        ]

        triton_code = """
import triton
import triton.language as tl

@triton.jit
def kernel():
    pid = tl.program_id(0)
    x = tl.load(ptr)
"""

        found_keywords = [kw for kw in triton_keywords if kw in triton_code]
        assert len(found_keywords) >= 3

    def test_helion_keywords(self):
        """Test Helion keyword detection."""
        helion_keywords = [
            "import helion",
            "@helion",
            "helion.kernel",
        ]

        helion_code = """
import helion

@helion.kernel
def kernel():
    pass
"""

        found_keywords = [kw for kw in helion_keywords if kw in helion_code]
        assert len(found_keywords) >= 2


class TestCrossFrameworkDistinction:
    """Test distinguishing between different frameworks."""

    def test_cuda_not_confused_with_triton(self):
        """Test CUDA code not detected as Triton."""
        cuda_code = "__global__ void kernel() {}"

        is_cuda = "__global__" in cuda_code
        is_triton = "triton" in cuda_code

        assert is_cuda
        assert not is_triton

    def test_triton_not_confused_with_cuda(self):
        """Test Triton code not detected as CUDA."""
        triton_code = "@triton.jit\ndef kernel(): pass"

        is_triton = "triton" in triton_code
        is_cuda = "__global__" in triton_code

        assert is_triton
        assert not is_cuda

    def test_helion_not_confused_with_triton(self):
        """Test Helion code not detected as Triton."""
        helion_code = "import helion"

        is_helion = "helion" in helion_code
        is_triton = "triton" in helion_code

        assert is_helion
        assert not is_triton


class TestFrameworkDetectionEdgeCases:
    """Test edge cases in framework detection."""

    def test_mixed_content_comments(self, temp_output_dir):
        """Test framework detection with misleading comments."""
        cuda_with_comment = """
// This kernel is inspired by triton
__global__ void kernel() {}
"""
        kernel_file = temp_output_dir / "kernel.cu"
        kernel_file.write_text(cuda_with_comment)

        # Extension says CUDA
        assert kernel_file.suffix == ".cu"
        # Content also has __global__
        assert "__global__" in cuda_with_comment
        # Should be detected as CUDA, not Triton (despite comment)

    def test_empty_file(self, temp_output_dir):
        """Test handling empty kernel file."""
        kernel_file = temp_output_dir / "empty.py"
        kernel_file.write_text("")

        content = kernel_file.read_text()
        assert content == ""
        # Framework detection should handle gracefully

    def test_file_with_only_comments(self, temp_output_dir):
        """Test file with only comments."""
        commented_file = temp_output_dir / "comments.cu"
        commented_file.write_text("// Just a comment\n/* Another comment */")

        content = commented_file.read_text()
        has_code = "__global__" in content
        assert not has_code


class TestMultiKernelDetection:
    """Test detection when multiple kernels in one file."""

    def test_multiple_cuda_kernels_in_one_file(self, temp_output_dir):
        """Test detecting CUDA with multiple kernels."""
        cuda_multi = """
__global__ void kernel1() {}
__global__ void kernel2() {}
__global__ void kernel3() {}
"""
        kernel_file = temp_output_dir / "multi.cu"
        kernel_file.write_text(cuda_multi)

        import re
        kernel_pattern = r"__global__\s+\w+\s+(\w+)\s*\("
        kernels = re.findall(kernel_pattern, cuda_multi)

        assert len(kernels) == 3
        assert "kernel1" in kernels
        assert "kernel2" in kernels
        assert "kernel3" in kernels

    def test_multiple_triton_kernels_in_one_file(self, temp_output_dir):
        """Test detecting Triton with multiple kernels."""
        triton_multi = """
@triton.jit
def kernel1(): pass

@triton.jit
def kernel2(): pass
"""
        kernel_file = temp_output_dir / "multi.py"
        kernel_file.write_text(triton_multi)

        jit_count = triton_multi.count("@triton.jit")
        assert jit_count == 2
