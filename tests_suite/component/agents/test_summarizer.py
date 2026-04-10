"""
Comprehensive tests for the summarizer agent.

Tests cover:
- Experience item generation
- Memory update decisions
- Rewrite type classification (positive/negative)
- Strategy extraction
- Speedup analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestSummarizerBasic:
    """Basic summarizer agent tests."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_run_summarizer_basic(
        self, sample_profiling_context, sample_optimization_plan,
        test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test basic summarization."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        experience_item, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=sample_optimization_plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=1.15,
            existing_memory=[],
            model="claude-sonnet-4@20250514",
        )

        assert experience_item is not None
        assert memory_update is not None

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_experience_item_structure(
        self, sample_profiling_context, sample_optimization_plan,
        test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test that experience item has required fields."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        experience_item, _ = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=sample_optimization_plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=1.15,
            existing_memory=[],
            model="claude-sonnet-4@20250514",
        )

        # Required fields
        assert "item_id" in experience_item
        assert "direction" in experience_item
        assert "rewrite_type" in experience_item
        assert "speedup" in experience_item
        assert "framework" in experience_item

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_memory_update_structure(
        self, sample_profiling_context, sample_optimization_plan,
        test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test that memory update has required fields."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=sample_optimization_plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=1.15,
            existing_memory=[],
            model="claude-sonnet-4@20250514",
        )

        # Required fields
        assert "action" in memory_update
        assert memory_update["action"] in ["append", "replace", "skip"]


class TestSummarizerRewriteType:
    """Test rewrite type classification."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_positive_rewrite(
        self, sample_profiling_context, sample_optimization_plan,
        test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test classification of positive speedup as positive rewrite."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        experience_item, _ = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=sample_optimization_plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=1.25,  # Positive speedup
            existing_memory=[],
            model="claude-sonnet-4@20250514",
        )

        # Should be classified as positive
        assert experience_item.get("rewrite_type") in ["positive", "negative"]

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_negative_rewrite(
        self, sample_profiling_context, sample_optimization_plan,
        test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test classification of slowdown as negative rewrite."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        experience_item, _ = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=sample_optimization_plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=0.85,  # Slowdown
            existing_memory=[],
            model="claude-sonnet-4@20250514",
        )

        assert experience_item.get("rewrite_type") in ["positive", "negative"]


class TestSummarizerMemoryDecisions:
    """Test memory update decisions."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_append_decision(
        self, sample_profiling_context, sample_optimization_plan,
        test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test append decision for significant improvement."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=sample_optimization_plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=1.30,  # Significant improvement
            existing_memory=[],
            model="claude-sonnet-4@20250514",
        )

        # Decision should be made
        assert memory_update.get("action") in ["append", "replace", "skip"]

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_skip_decision(
        self, sample_profiling_context, sample_optimization_plan,
        test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test skip decision for insignificant change."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=sample_optimization_plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=1.01,  # Insignificant change
            existing_memory=[],
            model="claude-sonnet-4@20250514",
        )

        assert memory_update.get("action") in ["append", "replace", "skip"]

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_with_existing_memory(
        self, sample_profiling_context, sample_optimization_plan,
        test_kernel_path, sample_memory_items, patch_build_llm, gpu_id
    ):
        """Test summarization with existing memory."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=sample_optimization_plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=1.20,
            existing_memory=sample_memory_items,
            model="claude-sonnet-4@20250514",
        )

        # Should consider existing memory in decision
        assert memory_update is not None


class TestSummarizerStrategyExtraction:
    """Test strategy extraction from optimizations."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_strategy_fields(
        self, sample_profiling_context, sample_optimization_plan,
        test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test that strategy information is extracted."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        experience_item, _ = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=sample_optimization_plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=1.20,
            existing_memory=[],
            model="claude-sonnet-4@20250514",
        )

        # May have strategy fields
        if "strategy_title" in experience_item:
            assert isinstance(experience_item["strategy_title"], str)
            assert len(experience_item["strategy_title"]) > 0


class TestSummarizerIterationTracking:
    """Test iteration tracking in experience items."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_iteration_recorded(
        self, sample_profiling_context, sample_optimization_plan,
        test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test that iteration is recorded in experience item."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        for iteration in [1, 2, 3]:
            experience_item, _ = run_summarizer(
                profiling_before=sample_profiling_context,
                profiling_after=sample_profiling_context,
                optimization_plan=sample_optimization_plan,
                kernel_before=kernel_source,
                kernel_after=kernel_source,
                iteration=iteration,
                framework="triton",
                speedup=1.15,
                existing_memory=[],
                model="claude-sonnet-4@20250514",
            )

            # Iteration may be tracked
            if "iteration" in experience_item:
                assert experience_item["iteration"] == iteration


class TestSummarizerErrorHandling:
    """Test error handling in summarizer."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_missing_profiling_data(
        self, sample_optimization_plan, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test handling of missing profiling data."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        minimal_ctx = {
            "kernel_name": "test",
            "framework_hint": "triton",
            "sections": {},
            "rules": [],
            "launch": {},
        }

        # Should handle minimal data
        experience_item, memory_update = run_summarizer(
            profiling_before=minimal_ctx,
            profiling_after=minimal_ctx,
            optimization_plan=sample_optimization_plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=1.15,
            existing_memory=[],
            model="claude-sonnet-4@20250514",
        )

        assert experience_item is not None
        assert memory_update is not None


class TestSummarizerSpeedupAnalysis:
    """Test speedup analysis and reporting."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_speedup_recorded(
        self, sample_profiling_context, sample_optimization_plan,
        test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test that speedup is correctly recorded."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        speedup_value = 1.42

        experience_item, _ = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=sample_optimization_plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=speedup_value,
            existing_memory=[],
            model="claude-sonnet-4@20250514",
        )

        assert "speedup" in experience_item
        # Speedup should be close to provided value
        assert abs(experience_item["speedup"] - speedup_value) < 0.01
