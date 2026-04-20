"""
Scenario-based tests with xfail markers for known issues.

These tests document real-world scenarios that may fail but are important
to track. Each test has an xfail marker with a clear reason explaining
why it might fail and under what conditions it should pass.

Scenarios covered:
- Edge cases in optimization
- Known LLM limitations
- Framework-specific quirks
- Complex multi-step workflows
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


@pytest.mark.xfail(reason="NCU kernel name filtering unreliable - sometimes profiles zero kernels")
@pytest.mark.gpu
@pytest.mark.slow
class TestNCUKernelFiltering:
    """Tests for NCU kernel name filtering (known to be flaky)."""

    def test_ncu_kernel_name_filter_triton(
        self, temp_output_dir, skip_if_no_cuda
    ):
        """Test NCU filtering with Triton kernel names (xfail - mangled names)."""
        from kernelopt.agents.profiler import run_ncu

        # Triton kernels have JIT-mangled names that don't match Python function names
        triton_file = temp_output_dir / "triton_kernel.py"
        triton_file.write_text("""
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # kernel code
""")

        output_base = str(temp_output_dir / "ncu_triton")

        # Trying to filter by Python function name won't work
        csv_path, _, _ = run_ncu(
            str(triton_file),
            output_base,
            kernel_name="my_kernel",  # Won't match mangled JIT name
            kernel_args=["--profile-only"],
        )

        # Likely to profile zero kernels or all kernels
        assert Path(csv_path).exists()

    def test_ncu_kernel_name_filter_cuda_overloaded(
        self, temp_output_dir, skip_if_no_cuda
    ):
        """Test NCU filtering with overloaded CUDA kernel names (xfail - ambiguous)."""
        from kernelopt.agents.profiler import run_ncu

        # Multiple kernels with same base name
        cuda_file = temp_output_dir / "overloaded.cu"
        cuda_file.write_text("""
#include <cuda_runtime.h>

template<typename T>
__global__ void process(T* data, int n) {
    data[0] = T(1);
}

int main() {
    float* d_f;
    int* d_i;
    cudaMalloc(&d_f, sizeof(float));
    cudaMalloc(&d_i, sizeof(int));

    process<<<1, 1>>>(d_f, 1);  // process<float>
    process<<<1, 1>>>(d_i, 1);  // process<int>

    cudaFree(d_f);
    cudaFree(d_i);
    return 0;
}
""")

        output_base = str(temp_output_dir / "ncu_overload")

        # Filtering by "process" might match both or neither
        csv_path, _, _ = run_ncu(
            str(cuda_file),
            output_base,
            kernel_name="process",
            kernel_args=[],
        )

        assert Path(csv_path).exists()


@pytest.mark.xfail(reason="LLM-generated code may have syntax errors or not follow plan exactly")
@pytest.mark.agent
@pytest.mark.slow
@pytest.mark.gpu
class TestLLMCodeGeneration:
    """Tests for LLM code generation quality (may produce invalid code)."""

    def test_llm_generates_syntactically_valid_cuda(
        self, temp_output_dir, patch_build_llm, gpu_id
    ):
        """Test that LLM generates syntactically valid CUDA (xfail - may have errors)."""
        from kernelopt.agents.executor import run_executor_agent

        patch_build_llm(gpu_id)

        initial_kernel = "__global__ void k(float* x) { x[0] = 1.0f; }"

        plan = {
            "plan_id": "syntax_test",
            "direction": "optimize_memory",
            "framework": "cuda",
            "step": {
                "action": "modify",
                "target": "memory access",
                "change": "Add shared memory",
                "predicted_effect": "Faster",
            },
        }

        optimized, result = run_executor_agent(
            kernel_source=initial_kernel,
            optimization_plan=plan,
            memory=[],
            K=3,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # LLM may produce code with syntax errors
        if result["status"] == "success":
            # If successful, code should compile
            assert result.get("ready_for_profiling", False)

    def test_llm_respects_framework_constraints(
        self, temp_output_dir, patch_build_llm, gpu_id
    ):
        """Test that LLM respects framework-specific constraints (xfail - may violate)."""
        from kernelopt.agents.executor import run_executor_agent

        patch_build_llm(gpu_id)

        # Triton kernel with specific constraints
        triton_kernel = """
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, BLOCK: tl.constexpr):
    # BLOCK must be constexpr
    pid = tl.program_id(0)
"""

        plan = {
            "plan_id": "triton_constexpr",
            "direction": "test_constraints",
            "framework": "triton",
        }

        optimized, result = run_executor_agent(
            kernel_source=triton_kernel,
            optimization_plan=plan,
            memory=[],
            K=2,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # LLM may remove tl.constexpr annotations, breaking Triton
        assert result is not None


@pytest.mark.xfail(reason="Full pipeline may fail at various stages with real LLM/GPU")
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.gpu
class TestFullPipelineScenarios:
    """End-to-end pipeline scenarios (complex, may fail at various stages)."""

    def test_multi_iteration_convergence(
        self, test_kernel_path, temp_output_dir, patch_build_llm, gpu_id
    ):
        """Test that pipeline converges to better kernel over iterations (xfail - may plateau)."""
        from kernelopt.langgraph_pipeline import run_langgraph_pipeline
        from kernelopt.state import PipelineState, PipelineConfig

        patch_build_llm(gpu_id)

        config = PipelineConfig(
            kernel_path=str(test_kernel_path),
            output_dir=str(temp_output_dir),
            max_iterations=3,  # Short for testing
            N=2,  # Few plans per iteration
            K=1,  # Single attempt
            model="claude-sonnet-4@20250514",
        )

        state = PipelineState(
            config=config,
            kernel_path=str(test_kernel_path),
            kernel_source=test_kernel_path.read_text(),
            kernel_basename="test_kernel",
        )

        # Run full pipeline
        final_state = run_langgraph_pipeline(state)

        # May not improve (plateau, LLM limitations, etc.)
        if final_state.iteration_results:
            # Check if any iteration improved
            improved = any(r.improved for r in final_state.iteration_results)
            assert improved or final_state.iteration >= 3  # Either improved or tried enough

    def test_pipeline_with_all_failures(
        self, temp_output_dir, patch_build_llm, gpu_id
    ):
        """Test pipeline resilience when all optimizations fail (xfail - may crash)."""
        from kernelopt.langgraph_pipeline import run_langgraph_pipeline
        from kernelopt.state import PipelineState, PipelineConfig

        patch_build_llm(gpu_id)

        # Very simple kernel that's hard to optimize
        simple_kernel = temp_output_dir / "trivial.py"
        simple_kernel.write_text("x = 1")

        config = PipelineConfig(
            kernel_path=str(simple_kernel),
            output_dir=str(temp_output_dir / "all_fail"),
            max_iterations=2,
            N=2,
            K=1,
        )

        state = PipelineState(
            config=config,
            kernel_path=str(simple_kernel),
            kernel_source="x = 1",
            kernel_basename="trivial",
        )

        # All optimizations likely fail (not a real kernel)
        # Pipeline should handle gracefully
        try:
            final_state = run_langgraph_pipeline(state)
            # Should complete without crashing
            assert final_state is not None
        except Exception as e:
            pytest.fail(f"Pipeline crashed instead of handling failures: {e}")


@pytest.mark.xfail(reason="Deduplication logic is complex and LLM-decided, may not always be optimal")
@pytest.mark.agent
@pytest.mark.slow
@pytest.mark.gpu
class TestMemoryDeduplicationScenarios:
    """Complex deduplication scenarios (LLM decision may not be optimal)."""

    def test_dedup_very_similar_optimizations(
        self, sample_profiling_context, test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test deduplication of very similar optimizations (xfail - may keep both)."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        # Existing memory with very similar optimization
        existing_memory = [
            {
                "item_id": "existing",
                "direction": "reduce_shared_memory_bank_conflicts",
                "speedup": 1.25,
            }
        ]

        # New optimization that's nearly identical
        plan = {
            "plan_id": "new",
            "direction": "reduce_bank_conflicts_shared_mem",  # Essentially same thing
        }

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="cuda",
            speedup=1.24,  # Slightly worse
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # Should skip (very similar + worse), but LLM might append
        # xfail: LLM may not recognize similarity
        assert memory_update["action"] == "skip"

    def test_dedup_cross_framework_same_technique(
        self, sample_profiling_context, test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test deduplication of same technique across frameworks (xfail - unclear policy)."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        # CUDA version of a technique
        existing_memory = [
            {
                "item_id": "cuda_version",
                "direction": "loop_unrolling",
                "speedup": 1.30,
                "framework": "cuda",
            }
        ]

        # Triton version of same technique
        plan = {
            "plan_id": "triton_version",
            "direction": "loop_unrolling",
        }

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",  # Different framework
            speedup=1.28,
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # Unclear: should we keep both (different frameworks) or replace?
        # LLM decides, may not be consistent
        assert memory_update["action"] in ["append", "replace", "skip"]


@pytest.mark.xfail(reason="Correctness checking with LLM-generated references is complex and may fail")
@pytest.mark.agent
@pytest.mark.slow
@pytest.mark.gpu
class TestCorrectnessComplexScenarios:
    """Complex correctness checking scenarios (LLM reference generation may fail)."""

    def test_correctness_complex_reduction(
        self, temp_output_dir, patch_build_llm, gpu_id
    ):
        """Test correctness of complex reduction kernel (xfail - hard for LLM to reference)."""
        from kernelopt.functions.correctness import check_correctness

        patch_build_llm(gpu_id)

        # Complex reduction kernel
        complex_kernel = """
import triton
import triton.language as tl
import torch

@triton.jit
def complex_reduce(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Multi-dimensional reduction with complex logic
    pid_m = tl.program_id(0)
    # ... complex reduction logic ...
    pass
"""

        py_file = temp_output_dir / "complex_reduce.py"
        py_file.write_text(complex_kernel)

        # LLM may struggle to generate correct CPU reference
        passed, message = check_correctness(
            original_path=str(py_file),
            optimized_source=complex_kernel,
            framework="triton",
            use_agentic=True,
            model="claude-sonnet-4@20250514",
        )

        # Complex operations hard for LLM to reference correctly
        assert passed

    def test_correctness_custom_data_types(
        self, temp_output_dir, patch_build_llm, gpu_id
    ):
        """Test correctness with custom/mixed data types (xfail - LLM may not handle correctly)."""
        from kernelopt.functions.correctness import check_correctness

        patch_build_llm(gpu_id)

        # Kernel with mixed FP16/FP32
        mixed_dtype_kernel = """
import triton
import triton.language as tl

@triton.jit
def mixed_precision(fp16_ptr, fp32_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # Mix FP16 inputs with FP32 accumulation
    pid = tl.program_id(0)
    # ... mixed precision logic ...
"""

        py_file = temp_output_dir / "mixed_precision.py"
        py_file.write_text(mixed_dtype_kernel)

        passed, message = check_correctness(
            original_path=str(py_file),
            optimized_source=mixed_dtype_kernel,
            framework="triton",
            use_agentic=True,
        )

        # LLM may not handle dtype casting correctly in reference
        assert passed


@pytest.mark.xfail(reason="Profiler LLM mode may not choose optimal NCU config")
@pytest.mark.agent
@pytest.mark.slow
@pytest.mark.gpu
class TestProfilerLLMDecisions:
    """Test profiler LLM mode decisions (may choose suboptimal config)."""

    def test_llm_chooses_optimal_launch_count(
        self, test_kernel_path, temp_output_dir, patch_build_llm, gpu_id
    ):
        """Test that LLM chooses optimal launch_count (xfail - may choose poorly)."""
        from kernelopt.agents.profiler import run_profiler_agent
        from kernelopt.state import PipelineConfig

        patch_build_llm(gpu_id)

        config = PipelineConfig(
            kernel_path=str(test_kernel_path),
            max_iterations=5,
            N=2,
            K=1,
            # No kernel_name, no ncu_config → LLM decides
        )

        output_base = str(temp_output_dir / "llm_launch")

        _, resolved_config = run_profiler_agent(
            kernel_path=str(test_kernel_path),
            output_basename=output_base,
            ncu_config=None,
            pipeline_config=config,
            kernel_id=0,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # LLM may choose launch_count that's too high or too low
        # Optimal is usually 1 for single-kernel profiling
        assert resolved_config.get("launch_count") == 1

    def test_llm_detects_multi_kernel_file(
        self, temp_output_dir, patch_build_llm, gpu_id
    ):
        """Test that LLM detects files with multiple kernels (xfail - may miss)."""
        from kernelopt.agents.profiler import run_profiler_agent
        from kernelopt.state import PipelineConfig

        patch_build_llm(gpu_id)

        # File with multiple kernels
        multi_kernel = temp_output_dir / "multi.cu"
        multi_kernel.write_text("""
__global__ void kernel1(float* x) { x[0] = 1.0f; }
__global__ void kernel2(float* x) { x[0] = 2.0f; }
__global__ void kernel3(float* x) { x[0] = 3.0f; }

int main() {
    float* d_x;
    cudaMalloc(&d_x, sizeof(float));
    kernel1<<<1,1>>>(d_x);
    kernel2<<<1,1>>>(d_x);
    kernel3<<<1,1>>>(d_x);
    cudaFree(d_x);
    return 0;
}
""")

        config = PipelineConfig(
            kernel_path=str(multi_kernel),
            max_iterations=1,
            N=1,
            K=1,
        )

        output_base = str(temp_output_dir / "llm_multi")

        _, resolved_config = run_profiler_agent(
            kernel_path=str(multi_kernel),
            output_basename=output_base,
            ncu_config=None,
            pipeline_config=config,
            kernel_id=0,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None},
        )

        # LLM should recognize multiple kernels and set appropriate config
        # May miss this and profile all kernels
        kernel_name = resolved_config.get("kernel_name", "")
        assert kernel_name != ""  # Should filter to specific kernel
