"""
Framework Utilities Tests.

Tests for framework detection and utility functions.
Tests helpers for identifying CUDA/Triton/Helion kernels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestFrameworkDetectionUtils:
    """Test framework detection utility functions."""

    def test_detect_cuda_by_extension_util(self):
        """Test utility to detect CUDA by file extension."""
        def is_cuda_extension(path: Path) -> bool:
            return path.suffix in [".cu", ".cuh"]

        assert is_cuda_extension(Path("kernel.cu"))
        assert is_cuda_extension(Path("header.cuh"))
        assert not is_cuda_extension(Path("kernel.py"))

    def test_detect_python_based_framework_util(self):
        """Test utility to detect Python-based frameworks."""
        def is_python_extension(path: Path) -> bool:
            return path.suffix == ".py"

        assert is_python_extension(Path("kernel.py"))
        assert not is_python_extension(Path("kernel.cu"))

    def test_framework_from_content_util(self, temp_output_dir):
        """Test utility to detect framework from content."""
        def detect_framework_from_content(content: str) -> str:
            if "__global__" in content:
                return "cuda"
            elif "import triton" in content or "@triton" in content:
                return "triton"
            elif "import helion" in content or "@helion" in content:
                return "helion"
            else:
                return "unknown"

        cuda_content = "__global__ void kernel() {}"
        triton_content = "import triton\n@triton.jit\ndef kernel(): pass"
        helion_content = "import helion\n@helion.kernel\ndef kernel(): pass"
        unknown_content = "def regular_function(): pass"

        assert detect_framework_from_content(cuda_content) == "cuda"
        assert detect_framework_from_content(triton_content) == "triton"
        assert detect_framework_from_content(helion_content) == "helion"
        assert detect_framework_from_content(unknown_content) == "unknown"


class TestCUDAKeywordDetection:
    """Test CUDA keyword detection utilities."""

    def test_detect_global_keyword(self):
        """Test detecting __global__ keyword."""
        def has_global_keyword(content: str) -> bool:
            return "__global__" in content

        assert has_global_keyword("__global__ void kernel() {}")
        assert not has_global_keyword("void regular_function() {}")

    def test_detect_device_keyword(self):
        """Test detecting __device__ keyword."""
        def has_device_keyword(content: str) -> bool:
            return "__device__" in content

        assert has_device_keyword("__device__ float helper() {}")
        assert not has_device_keyword("float regular_helper() {}")

    def test_detect_shared_memory(self):
        """Test detecting __shared__ keyword."""
        def has_shared_memory(content: str) -> bool:
            return "__shared__" in content

        cuda_shared = "__shared__ float s_data[256];"
        assert has_shared_memory(cuda_shared)

    def test_detect_kernel_launch(self):
        """Test detecting kernel launch syntax."""
        def has_kernel_launch(content: str) -> bool:
            return "<<<" in content and ">>>" in content

        kernel_launch = "kernel<<<blocks, threads>>>(args);"
        assert has_kernel_launch(kernel_launch)


class TestTritonKeywordDetection:
    """Test Triton keyword detection utilities."""

    def test_detect_triton_import(self):
        """Test detecting Triton import."""
        def has_triton_import(content: str) -> bool:
            return "import triton" in content

        assert has_triton_import("import triton")
        assert has_triton_import("import triton.language as tl")
        assert not has_triton_import("# triton comment")

    def test_detect_triton_jit(self):
        """Test detecting @triton.jit decorator."""
        def has_triton_jit(content: str) -> bool:
            return "@triton.jit" in content

        assert has_triton_jit("@triton.jit\ndef kernel(): pass")
        assert not has_triton_jit("def kernel(): pass")

    def test_detect_triton_language_ops(self):
        """Test detecting triton.language operations."""
        def has_tl_ops(content: str) -> bool:
            tl_keywords = ["tl.load", "tl.store", "tl.program_id", "tl.arange"]
            return any(kw in content for kw in tl_keywords)

        assert has_tl_ops("x = tl.load(ptr)")
        assert has_tl_ops("pid = tl.program_id(0)")
        assert not has_tl_ops("regular python code")


class TestHelionKeywordDetection:
    """Test Helion keyword detection utilities."""

    def test_detect_helion_import(self):
        """Test detecting Helion import."""
        def has_helion_import(content: str) -> bool:
            return "import helion" in content

        assert has_helion_import("import helion")
        assert not has_helion_import("# helion comment")

    def test_detect_helion_decorator(self):
        """Test detecting @helion decorator."""
        def has_helion_decorator(content: str) -> bool:
            return "@helion" in content

        assert has_helion_decorator("@helion.kernel\ndef kernel(): pass")
        assert not has_helion_decorator("def kernel(): pass")


class TestFrameworkPriority:
    """Test framework detection priority utilities."""

    def test_extension_priority(self, temp_output_dir):
        """Test extension-based detection priority."""
        def detect_framework(path: Path, content: str) -> str:
            # Extension takes priority
            if path.suffix in [".cu", ".cuh"]:
                return "cuda"
            elif path.suffix == ".py":
                # Check content for Python-based frameworks
                if "import triton" in content:
                    return "triton"
                elif "import helion" in content:
                    return "helion"
                else:
                    return "unknown"
            return "unknown"

        cu_file = temp_output_dir / "kernel.cu"
        py_triton = temp_output_dir / "kernel.py"

        cu_file.write_text("any content")
        py_triton.write_text("import triton")

        assert detect_framework(cu_file, cu_file.read_text()) == "cuda"
        assert detect_framework(py_triton, py_triton.read_text()) == "triton"


class TestKernelNameExtraction:
    """Test kernel name extraction utilities."""

    def test_extract_cuda_kernel_names(self):
        """Test extracting CUDA kernel names."""
        import re

        cuda_source = """
__global__ void kernel_one(float* x) {}
__global__ void kernel_two(float* y) {}
__device__ void helper() {}
"""

        kernel_pattern = r"__global__\s+\w+\s+(\w+)\s*\("
        kernels = re.findall(kernel_pattern, cuda_source)

        assert "kernel_one" in kernels
        assert "kernel_two" in kernels
        assert "helper" not in kernels  # __device__, not __global__

    def test_extract_triton_kernel_names(self):
        """Test extracting Triton kernel names."""
        import re

        triton_source = """
@triton.jit
def kernel_one(): pass

@triton.jit
def kernel_two(): pass

def regular_function(): pass
"""

        # Pattern: @triton.jit followed by def <name>
        kernel_pattern = r"@triton\.jit\s+def\s+(\w+)"
        kernels = re.findall(kernel_pattern, triton_source)

        assert "kernel_one" in kernels
        assert "kernel_two" in kernels
        assert "regular_function" not in kernels


class TestFilePathUtils:
    """Test file path utility functions."""

    def test_get_kernel_basename(self):
        """Test getting kernel base name from path."""
        def get_basename(path: Path) -> str:
            return path.stem

        assert get_basename(Path("/path/to/kernel.cu")) == "kernel"
        assert get_basename(Path("kernel.py")) == "kernel"

    def test_get_output_path(self, temp_output_dir):
        """Test generating output file paths."""
        def get_output_path(base_dir: Path, basename: str, suffix: str) -> Path:
            return base_dir / f"{basename}{suffix}"

        output_path = get_output_path(temp_output_dir, "kernel", ".csv")
        assert output_path.name == "kernel.csv"
        assert output_path.parent == temp_output_dir

    def test_ensure_output_dir_exists(self, temp_output_dir):
        """Test ensuring output directory exists."""
        def ensure_dir(path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)

        new_dir = temp_output_dir / "subdir"
        assert not new_dir.exists()

        ensure_dir(new_dir)
        assert new_dir.exists()


class TestFrameworkCompatibility:
    """Test framework compatibility utilities."""

    def test_cuda_requires_gpu(self):
        """Test CUDA requires GPU environment."""
        def requires_gpu(framework: str) -> bool:
            return framework in ["cuda", "triton", "helion"]

        assert requires_gpu("cuda")
        assert requires_gpu("triton")
        assert not requires_gpu("python")

    def test_framework_supports_ncu(self):
        """Test which frameworks support NCU profiling."""
        def supports_ncu(framework: str) -> bool:
            # CUDA directly supports NCU
            # Triton compiles to CUDA, so may support NCU
            # Helion may or may not support NCU
            return framework in ["cuda", "triton"]

        assert supports_ncu("cuda")
        assert supports_ncu("triton")
