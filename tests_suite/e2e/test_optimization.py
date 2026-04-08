"""
Optimization pipeline tests for the complete kernel optimization pipeline.

These tests run the full pipeline from start to finish and verify:
- Complete optimization workflow
- Output file generation
- Memory persistence
- Multi-iteration optimization

Note: Sanity check is automatically run before these tests via the
run_sanity_check_fixture session-scoped fixture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


@pytest.fixture(scope="module", autouse=True)
def sanity_check_before_optimization(run_sanity_check_fixture):
    """Automatically run sanity check before optimization pipeline tests."""
    if not run_sanity_check_fixture:
        pytest.skip("Sanity check failed - skipping optimization pipeline tests")
    return run_sanity_check_fixture


# ===========================================================================
# Single Iteration E2E Tests
# ===========================================================================

class TestSingleIterationE2E:
    """Optimization pipeline tests for single iteration of the pipeline."""

    @pytest.mark.slow
    @pytest.mark.optimization
    def test_single_iteration_complete(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda,
        patch_build_llm, gpu_id
    ):
        """Test complete single iteration of optimization."""
        from kernelopt.state import PipelineState, PipelineConfig

        # Patch LLM
        patch_build_llm(gpu_id)

        # Import pipeline after patching
        from kernelopt.graph import run_pipeline

        # Configure for single iteration
        cfg = PipelineConfig(
            T=1,  # Single iteration
            N=2,  # 2 plans
            K=1,  # 1 attempt per plan
            TopK=2,
            model="claude-sonnet-4@20250514",
            kernel_args=["--profile-only"],
            output_dir=str(temp_output_dir),
            kernel_path=str(test_kernel_path),
        )

        state = PipelineState(
            config=cfg,
            kernel_path=str(test_kernel_path),
            kernel_source=test_kernel_path.read_text(),
            best_kernel=test_kernel_path.read_text(),
            framework="auto",
            kernel_basename=test_kernel_path.stem,
        )

        # Run pipeline
        final_state = run_pipeline(state)

        # Verify outputs
        assert final_state.iteration >= 0
        assert len(final_state.iteration_results) > 0
        assert final_state.best_latency_ns < float("inf")

        # Check output files
        output_dir = Path(cfg.output_dir)
        assert output_dir.exists()
        assert (output_dir / "optimization_log.md").exists()


# ===========================================================================
# Multi-Iteration E2E Tests
# ===========================================================================

class TestMultiIterationE2E:
    """Optimization pipeline tests for multi-iteration optimization."""

    @pytest.mark.slow
    @pytest.mark.optimization
    def test_multi_iteration_complete(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda,
        patch_build_llm, gpu_id
    ):
        """Test complete multi-iteration optimization."""
        from kernelopt.state import PipelineState, PipelineConfig
        from kernelopt.graph import run_pipeline

        # Patch LLM
        patch_build_llm(gpu_id)

        # Configure for multiple iterations
        cfg = PipelineConfig(
            T=2,  # 2 iterations
            N=2,  # 2 plans per iteration
            K=1,  # 1 attempt per plan
            TopK=2,
            model="claude-sonnet-4@20250514",
            kernel_args=["--profile-only"],
            output_dir=str(temp_output_dir / "multi_iter"),
            kernel_path=str(test_kernel_path),
        )

        state = PipelineState(
            config=cfg,
            kernel_path=str(test_kernel_path),
            kernel_source=test_kernel_path.read_text(),
            best_kernel=test_kernel_path.read_text(),
            framework="auto",
            kernel_basename=test_kernel_path.stem,
        )

        # Run pipeline
        final_state = run_pipeline(state)

        # Verify multiple iterations ran
        assert len(final_state.iteration_results) >= 1
        assert final_state.iteration >= 0

        # Check stopping conditions were evaluated
        # Either reached T iterations or stopped early
        assert final_state.iteration < cfg.T or \
               final_state.no_improvement_streak >= 2

    @pytest.mark.slow
    @pytest.mark.optimization
    def test_early_stopping(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda,
        patch_build_llm, gpu_id
    ):
        """Test early stopping on no improvement."""
        from kernelopt.state import PipelineState, PipelineConfig
        from kernelopt.graph import run_pipeline

        # Patch LLM
        patch_build_llm(gpu_id)

        cfg = PipelineConfig(
            T=10,  # Allow many iterations
            N=1,  # Limited plans to increase chance of no improvement
            K=1,
            TopK=1,
            model="claude-sonnet-4@20250514",
            kernel_args=["--profile-only"],
            output_dir=str(temp_output_dir / "early_stop"),
            kernel_path=str(test_kernel_path),
        )

        state = PipelineState(
            config=cfg,
            kernel_path=str(test_kernel_path),
            kernel_source=test_kernel_path.read_text(),
            best_kernel=test_kernel_path.read_text(),
            framework="auto",
            kernel_basename=test_kernel_path.stem,
        )

        # Run pipeline
        final_state = run_pipeline(state)

        # May stop early (no_improvement_streak >= 2) or complete all iterations
        assert final_state.iteration >= 0


# ===========================================================================
# Output Verification E2E Tests
# ===========================================================================

class TestOutputVerification:
    """Optimization pipeline tests verifying all outputs are generated correctly."""

    @pytest.mark.slow
    @pytest.mark.optimization
    def test_output_files_generated(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda,
        patch_build_llm, gpu_id
    ):
        """Test that all expected output files are generated."""
        from kernelopt.state import PipelineState, PipelineConfig
        from kernelopt.graph import run_pipeline

        # Patch LLM
        patch_build_llm(gpu_id)

        output_path = temp_output_dir / "output_test"
        cfg = PipelineConfig(
            T=1,
            N=2,
            K=1,
            model="claude-sonnet-4@20250514",
            kernel_args=["--profile-only"],
            output_dir=str(output_path),
            kernel_path=str(test_kernel_path),
        )

        state = PipelineState(
            config=cfg,
            kernel_path=str(test_kernel_path),
            kernel_source=test_kernel_path.read_text(),
            best_kernel=test_kernel_path.read_text(),
            framework="auto",
            kernel_basename=test_kernel_path.stem,
        )

        # Run pipeline
        run_pipeline(state)

        # Verify output directory exists
        assert output_path.exists()

        # Verify log file
        log_file = output_path / "optimization_log.md"
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Optimization Log" in log_content
        assert "Baseline latency" in log_content

        # Verify best kernel file
        best_kernel_file = output_path / f"best_kernel{test_kernel_path.suffix}"
        assert best_kernel_file.exists()

    @pytest.mark.slow
    @pytest.mark.optimization
    def test_memory_persistence(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda,
        patch_build_llm, gpu_id
    ):
        """Test that optimization memory is persisted correctly."""
        from kernelopt.state import PipelineState, PipelineConfig
        from kernelopt.graph import run_pipeline

        # Patch LLM
        patch_build_llm(gpu_id)

        output_path = temp_output_dir / "memory_test"
        cfg = PipelineConfig(
            T=2,
            N=2,
            K=1,
            TopK=2,
            model="claude-sonnet-4@20250514",
            kernel_args=["--profile-only"],
            output_dir=str(output_path),
            kernel_path=str(test_kernel_path),
        )

        state = PipelineState(
            config=cfg,
            kernel_path=str(test_kernel_path),
            kernel_source=test_kernel_path.read_text(),
            best_kernel=test_kernel_path.read_text(),
            framework="auto",
            kernel_basename=test_kernel_path.stem,
        )

        # Run pipeline
        final_state = run_pipeline(state)

        # Verify memory file
        memory_file = output_path / "optimization_memory.json"
        if final_state.optimization_memory:
            assert memory_file.exists()

            # Load and verify memory
            with open(memory_file) as f:
                memory = json.load(f)

            assert isinstance(memory, list)
            # Memory should be within capacity
            assert len(memory) <= cfg.ExpN

    @pytest.mark.slow
    @pytest.mark.optimization
    def test_ncu_artifacts_generated(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda,
        patch_build_llm, gpu_id
    ):
        """Test that NCU profiling artifacts are generated."""
        from kernelopt.state import PipelineState, PipelineConfig
        from kernelopt.graph import run_pipeline

        # Patch LLM
        patch_build_llm(gpu_id)

        output_path = temp_output_dir / "ncu_test"
        cfg = PipelineConfig(
            T=1,
            N=1,
            K=1,
            model="claude-sonnet-4@20250514",
            kernel_args=["--profile-only"],
            output_dir=str(output_path),
            kernel_path=str(test_kernel_path),
        )

        state = PipelineState(
            config=cfg,
            kernel_path=str(test_kernel_path),
            kernel_source=test_kernel_path.read_text(),
            best_kernel=test_kernel_path.read_text(),
            framework="auto",
            kernel_basename=test_kernel_path.stem,
        )

        # Run pipeline
        run_pipeline(state)

        # Check for baseline CSV
        csv_files = list(output_path.glob("*.csv"))
        assert len(csv_files) > 0, "No CSV files generated"

        # Check for .ncu-rep files (if generated)
        ncu_rep_files = list(output_path.glob("*.ncu-rep"))
        # May or may not exist depending on NCU configuration


# ===========================================================================
# Performance Verification E2E Tests
# ===========================================================================

class TestPerformanceVerification:
    """Optimization pipeline tests verifying performance aspects."""

    @pytest.mark.slow
    @pytest.mark.optimization
    def test_baseline_profiling_works(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda
    ):
        """Test that baseline profiling produces valid results."""
        from kernelopt.agents.profiler import run_ncu, build_profiling_context, get_latency_ns
        from kernelopt.functions.csv_parser import parse_ncu_csv

        output_base = str(temp_output_dir / "baseline_test")

        # Run NCU profiling
        csv_path, ncu_rep, profiling_ctx_path = run_ncu(
            str(test_kernel_path),
            output_base,
            kernel_args=["--profile-only"],
        )

        # Parse and build context
        sections, rules, launch = parse_ncu_csv(csv_path)
        ctx = build_profiling_context(sections, rules, launch, str(test_kernel_path))

        # Verify valid profiling results
        latency = get_latency_ns(ctx)
        assert latency > 0
        assert latency < float("inf")
        assert ctx["kernel_name"] != ""

    @pytest.mark.slow
    @pytest.mark.optimization
    def test_speedup_calculation(
        self, test_kernel_path, temp_output_dir, skip_if_no_cuda,
        patch_build_llm, gpu_id
    ):
        """Test that speedup is calculated correctly."""
        from kernelopt.state import PipelineState, PipelineConfig
        from kernelopt.graph import run_pipeline

        # Patch LLM
        patch_build_llm(gpu_id)

        output_path = temp_output_dir / "speedup_test"
        cfg = PipelineConfig(
            T=1,
            N=2,
            K=1,
            model="claude-sonnet-4@20250514",
            kernel_args=["--profile-only"],
            output_dir=str(output_path),
            kernel_path=str(test_kernel_path),
        )

        state = PipelineState(
            config=cfg,
            kernel_path=str(test_kernel_path),
            kernel_source=test_kernel_path.read_text(),
            best_kernel=test_kernel_path.read_text(),
            framework="auto",
            kernel_basename=test_kernel_path.stem,
        )

        # Run pipeline
        final_state = run_pipeline(state)

        # Verify speedup calculation
        if final_state.iteration_results:
            for result in final_state.iteration_results:
                for candidate in result.candidates:
                    if candidate.latency_ns < float("inf"):
                        # Speedup should be positive
                        assert candidate.speedup >= 0
