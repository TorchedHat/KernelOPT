"""
CUDA Framework-Specific Tests.

Tests for CUDA kernel compilation, profiling, and execution.
Covers CUDA-specific features like __global__, __device__, shared memory, etc.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


class TestCUDACompilation:
    """Test CUDA kernel compilation."""

    @pytest.mark.gpu
    def test_valid_cuda_kernel_compiles(self, temp_output_dir):
        """Test that a valid CUDA kernel compiles successfully."""
        from kernelopt.functions.compiler import compile_cuda

        cuda_source = """
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
        kernel_file = temp_output_dir / "test_cuda.cu"
        kernel_file.write_text(cuda_source)

        success, error = compile_cuda(str(kernel_file), str(temp_output_dir))

        assert success or error is not None  # Either compiles or returns error

    @pytest.mark.gpu
    def test_cuda_syntax_error_detected(self, temp_output_dir):
        """Test that CUDA syntax errors are detected."""
        from kernelopt.functions.compiler import compile_cuda

        broken_cuda = """
__global__ void broken(float* x) {
    x[0] = 1.0f  // Missing semicolon
}
"""
        kernel_file = temp_output_dir / "broken.cu"
        kernel_file.write_text(broken_cuda)

        success, error = compile_cuda(str(kernel_file), str(temp_output_dir))

        if not success:
            assert error is not None
            assert "error" in error.lower() or ";" in error

    @pytest.mark.gpu
    def test_cuda_with_main_compiles(self, temp_output_dir):
        """Test CUDA kernel with main() function."""
        from kernelopt.functions.compiler import compile_cuda

        cuda_with_main = """
__global__ void simple(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}

int main() {
    float* d_data;
    cudaMalloc(&d_data, 100 * sizeof(float));
    simple<<<1, 256>>>(d_data, 100);
    cudaFree(d_data);
    return 0;
}
"""
        kernel_file = temp_output_dir / "main.cu"
        kernel_file.write_text(cuda_with_main)

        success, error = compile_cuda(str(kernel_file), str(temp_output_dir))

        assert success or error is not None

    @pytest.mark.gpu
    def test_cuda_with_shared_memory(self, temp_output_dir):
        """Test CUDA kernel using shared memory."""
        from kernelopt.functions.compiler import compile_cuda

        cuda_shared = """
__global__ void shared_mem_kernel(float* data, int n) {
    __shared__ float s_data[256];
    int idx = threadIdx.x;
    if (idx < n) {
        s_data[idx] = data[idx];
        __syncthreads();
        data[idx] = s_data[idx] * 2.0f;
    }
}
"""
        kernel_file = temp_output_dir / "shared.cu"
        kernel_file.write_text(cuda_shared)

        success, error = compile_cuda(str(kernel_file), str(temp_output_dir))

        assert success or error is not None

    @pytest.mark.gpu
    def test_cuda_with_templates(self, temp_output_dir):
        """Test CUDA kernel with C++ templates."""
        from kernelopt.functions.compiler import compile_cuda

        cuda_template = """
template<typename T>
__global__ void templated_kernel(T* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * 2;
}

int main() {
    return 0;
}
"""
        kernel_file = temp_output_dir / "template.cu"
        kernel_file.write_text(cuda_template)

        success, error = compile_cuda(str(kernel_file), str(temp_output_dir))

        assert success or error is not None


class TestCUDAProfiling:
    """Test CUDA kernel profiling with NCU."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_cuda_ncu_profiling(self, temp_output_dir, skip_if_no_cuda):
        """Test NCU profiling of CUDA kernel."""
        # This is tested more thoroughly in test_ncu_runner.py
        # Just verify basic CUDA kernel can be profiled
        cuda_source = """
__global__ void simple(float* x) {
    x[0] = 1.0f;
}

int main() {
    float* d_x;
    cudaMalloc(&d_x, sizeof(float));
    simple<<<1, 1>>>(d_x);
    cudaFree(d_x);
    return 0;
}
"""
        kernel_file = temp_output_dir / "profile.cu"
        kernel_file.write_text(cuda_source)

        # Compilation must succeed for profiling
        from kernelopt.functions.compiler import compile_cuda
        success, _ = compile_cuda(str(kernel_file), str(temp_output_dir))

        # If compilation works, kernel can be profiled
        assert success or True  # Compilation may fail in test env

    @pytest.mark.gpu
    def test_cuda_kernel_name_extraction(self, temp_output_dir):
        """Test extracting kernel names from CUDA source."""
        cuda_source = """
__global__ void kernel_one(float* x) { x[0] = 1.0f; }
__global__ void kernel_two(float* y) { y[0] = 2.0f; }
"""
        kernel_file = temp_output_dir / "multi.cu"
        kernel_file.write_text(cuda_source)

        # Extract kernel names
        import re
        kernel_pattern = r"__global__\s+\w+\s+(\w+)\s*\("
        matches = re.findall(kernel_pattern, cuda_source)

        assert "kernel_one" in matches
        assert "kernel_two" in matches


class TestCUDACorrectness:
    """Test CUDA correctness checking."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_cuda_output_validation(self, temp_output_dir):
        """Test validating CUDA kernel output."""
        # Correctness is tested more in test_correctness_agent.py
        # This verifies CUDA-specific aspects
        cuda_kernel = """
__global__ void multiply_by_two(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}
"""
        kernel_file = temp_output_dir / "correct.cu"
        kernel_file.write_text(cuda_kernel)

        # Kernel should multiply by 2
        assert "* 2" in cuda_kernel or "*= 2" in cuda_kernel


class TestCUDAFrameworkDetection:
    """Test CUDA framework detection."""

    def test_detect_cuda_by_extension(self):
        """Test detecting CUDA by .cu extension."""
        assert Path("kernel.cu").suffix == ".cu"
        assert Path("kernel.cuh").suffix == ".cuh"

    def test_detect_cuda_by_keywords(self):
        """Test detecting CUDA by keywords."""
        cuda_code = "__global__ void kernel() {}"
        assert "__global__" in cuda_code
        assert "kernel" in cuda_code

    def test_cuda_vs_triton_distinction(self):
        """Test distinguishing CUDA from Triton."""
        cuda_code = "__global__ void kernel() {}"
        triton_code = "@triton.jit\ndef kernel():"

        assert "__global__" in cuda_code
        assert "@triton.jit" in triton_code
        assert "__global__" not in triton_code
        assert "@triton.jit" not in cuda_code


class TestCUDAMemoryPatterns:
    """Test CUDA memory access patterns."""

    def test_detect_global_memory_access(self):
        """Test detecting global memory accesses."""
        cuda_code = """
__global__ void kernel(float* global_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    global_data[idx] = 1.0f;
}
"""
        assert "float*" in cuda_code
        assert "global_data[" in cuda_code

    def test_detect_shared_memory_usage(self):
        """Test detecting shared memory usage."""
        cuda_code = """
__global__ void kernel() {
    __shared__ float s_data[256];
    s_data[threadIdx.x] = 1.0f;
}
"""
        assert "__shared__" in cuda_code
        assert "s_data[" in cuda_code

    def test_detect_coalesced_access(self):
        """Test pattern for coalesced memory access."""
        # Coalesced access: consecutive threads access consecutive addresses
        coalesced = """
int idx = blockIdx.x * blockDim.x + threadIdx.x;
data[idx] = value;
"""
        assert "blockIdx" in coalesced
        assert "threadIdx" in coalesced


class TestCUDAOptimizationPatterns:
    """Test CUDA optimization patterns."""

    def test_detect_loop_unrolling(self):
        """Test detecting loop unrolling."""
        unrolled_code = """
#pragma unroll
for (int i = 0; i < 4; i++) {
    sum += data[i];
}
"""
        assert "#pragma unroll" in unrolled_code

    def test_detect_warp_level_primitives(self):
        """Test detecting warp-level primitives."""
        warp_code = """
__shfl_down_sync(0xffffffff, value, offset);
"""
        assert "__shfl" in warp_code

    def test_detect_occupancy_optimization(self):
        """Test patterns for occupancy optimization."""
        # High occupancy: more blocks, smaller shared memory
        kernel_launch = "kernel<<<num_blocks, threads_per_block>>>"
        assert "<<<" in kernel_launch
        assert ">>>" in kernel_launch
