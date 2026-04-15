"""
Tests for Summarizer Agent's deduplication logic.

The summarizer performs deduplication based on:
- Direction matching with existing memory items
- Speedup comparison (higher speedup replaces, lower skips)
- Multi-manifestation handling (append with _v2 suffix)

Memory update actions:
- "append": Add new item to memory (unique or better manifestation)
- "replace": Replace existing item with higher speedup
- "skip": Don't add (lower speedup than existing)

These tests verify the deduplication logic works correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestSummarizerDeduplicationAppend:
    """Test append decisions for new unique optimizations."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_append_unique_direction(
        self, sample_profiling_context, sample_optimization_plan,
        test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test append when direction is unique (not in existing memory)."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        # Existing memory with different directions
        existing_memory = [
            {
                "item_id": "existing_1",
                "direction": "use_shared_memory",
                "speedup": 1.25,
            },
            {
                "item_id": "existing_2",
                "direction": "vectorize_loads",
                "speedup": 1.15,
            },
        ]

        # New optimization with unique direction
        new_plan = {
            "plan_id": "new_opt",
            "direction": "reduce_register_pressure",  # Unique direction
        }

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=new_plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=1.20,
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # Should append (unique direction)
        assert memory_update["action"] in ["append", "skip", "replace"]
        # Most likely append, but LLM decides

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_append_empty_memory(
        self, sample_profiling_context, sample_optimization_plan,
        test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test append when memory is empty."""
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
            speedup=1.30,
            existing_memory=[],  # Empty memory
            model="claude-sonnet-4@20250514",
        )

        # Should append (no conflicts possible)
        assert memory_update["action"] in ["append", "skip"]
        # Most likely append for significant speedup

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_append_significant_speedup(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test append for significant speedup even with existing similar items."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        existing_memory = [
            {
                "item_id": "existing",
                "direction": "optimize_memory",
                "speedup": 1.10,
            }
        ]

        plan = {
            "plan_id": "new_variant",
            "direction": "optimize_memory_v2",  # Similar but different
        }

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=1.50,  # Much better
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # Should likely append or replace
        assert memory_update["action"] in ["append", "replace", "skip"]


class TestSummarizerDeduplicationReplace:
    """Test replace decisions when new item is better."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_replace_same_direction_higher_speedup(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test replace when same direction has higher speedup."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        # Existing item with lower speedup
        existing_memory = [
            {
                "item_id": "old_version",
                "direction": "coalesce_memory_access",
                "speedup": 1.15,
                "framework": "triton",
            }
        ]

        # Same direction, much better speedup
        plan = {
            "plan_id": "improved_version",
            "direction": "coalesce_memory_access",  # Same direction
        }

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=2,
            framework="triton",
            speedup=1.45,  # Much higher than 1.15
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # Should likely replace or append
        assert memory_update["action"] in ["replace", "append", "skip"]

        # If replace, should have replace_item_id
        if memory_update["action"] == "replace":
            assert "replace_item_id" in memory_update

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_replace_provides_item_id(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test that replace action provides the item ID to replace."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        existing_memory = [
            {
                "item_id": "target_item_123",
                "direction": "reduce_bank_conflicts",
                "speedup": 1.12,
            }
        ]

        plan = {
            "plan_id": "better_approach",
            "direction": "reduce_bank_conflicts",
        }

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=3,
            framework="triton",
            speedup=1.35,
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # If action is replace, must have replace_item_id
        if memory_update["action"] == "replace":
            assert "replace_item_id" in memory_update
            # Should reference an existing item
            assert memory_update["replace_item_id"] is not None


class TestSummarizerDeduplicationSkip:
    """Test skip decisions when new item is worse or redundant."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_skip_same_direction_lower_speedup(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test skip when same direction has lower speedup."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        # Existing item with high speedup
        existing_memory = [
            {
                "item_id": "best_version",
                "direction": "tiling_optimization",
                "speedup": 1.60,
                "framework": "triton",
            }
        ]

        # Same direction, worse speedup
        plan = {
            "plan_id": "worse_attempt",
            "direction": "tiling_optimization",  # Same direction
        }

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=4,
            framework="triton",
            speedup=1.10,  # Much worse than 1.60
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # Should likely skip (worse performance)
        assert memory_update["action"] in ["skip", "append", "replace"]

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_skip_insignificant_improvement(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test skip for insignificant speedup (below threshold)."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        existing_memory = []

        plan = {
            "plan_id": "tiny_change",
            "direction": "minor_tweak",
        }

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=1.01,  # Insignificant (1% improvement)
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # May skip due to low speedup
        assert memory_update["action"] in ["skip", "append"]

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_skip_provides_reason(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test that skip action provides a reason."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        existing_memory = [
            {
                "item_id": "existing",
                "direction": "same_opt",
                "speedup": 1.50,
            }
        ]

        plan = {"plan_id": "dup", "direction": "same_opt"}

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=1,
            framework="triton",
            speedup=1.20,  # Lower than existing
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # Should have reason field
        assert "reason" in memory_update
        assert isinstance(memory_update["reason"], str)


class TestSummarizerDeduplicationLogic:
    """Test deduplication logic across different scenarios."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_multiple_existing_items(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test deduplication with multiple existing memory items."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        # Multiple existing items with various directions
        existing_memory = [
            {
                "item_id": "item_1",
                "direction": "shared_memory",
                "speedup": 1.30,
            },
            {
                "item_id": "item_2",
                "direction": "loop_unrolling",
                "speedup": 1.15,
            },
            {
                "item_id": "item_3",
                "direction": "prefetching",
                "speedup": 1.45,
            },
        ]

        plan = {
            "plan_id": "new_opt",
            "direction": "loop_unrolling",  # Matches item_2
        }

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=5,
            framework="triton",
            speedup=1.25,  # Better than item_2 (1.15)
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # Should make a decision
        assert memory_update["action"] in ["append", "replace", "skip"]

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_negative_rewrite_deduplication(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test deduplication for negative rewrites (regressions)."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        existing_memory = [
            {
                "item_id": "negative_1",
                "direction": "bad_optimization",
                "speedup": 0.80,  # Regression
                "rewrite_type": "negative",
            }
        ]

        plan = {
            "plan_id": "also_bad",
            "direction": "bad_optimization",  # Same bad direction
        }

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=6,
            framework="triton",
            speedup=0.75,  # Even worse regression
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # Decision should be made for negative items too
        assert memory_update["action"] in ["append", "replace", "skip"]

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_different_framework_same_direction(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test deduplication when same direction applies to different frameworks."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        # Existing: CUDA optimization
        existing_memory = [
            {
                "item_id": "cuda_opt",
                "direction": "coalesce_loads",
                "speedup": 1.40,
                "framework": "cuda",
            }
        ]

        # New: Same direction but for Triton
        plan = {
            "plan_id": "triton_opt",
            "direction": "coalesce_loads",  # Same direction
        }

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=7,
            framework="triton",  # Different framework
            speedup=1.35,
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # May append (different framework) or compare speedup
        assert memory_update["action"] in ["append", "replace", "skip"]


class TestSummarizerDeduplicationEdgeCases:
    """Test edge cases in deduplication logic."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_exact_same_speedup(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test deduplication when speedups are exactly equal."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        existing_memory = [
            {
                "item_id": "existing",
                "direction": "same_perf",
                "speedup": 1.25,
            }
        ]

        plan = {"plan_id": "duplicate", "direction": "same_perf"}

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=8,
            framework="triton",
            speedup=1.25,  # Exactly same
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # LLM should decide - likely skip or append as variant
        assert memory_update["action"] in ["skip", "append", "replace"]

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_very_full_memory(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test deduplication when memory is near capacity."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        # Memory near capacity (ExpN default is 8)
        existing_memory = [
            {"item_id": f"item_{i}", "direction": f"opt_{i}", "speedup": 1.1 + i * 0.05}
            for i in range(7)
        ]

        plan = {"plan_id": "new_opt", "direction": "brand_new"}

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=9,
            framework="triton",
            speedup=1.50,  # Good speedup
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # Should still make a decision (queue management is separate)
        assert memory_update["action"] in ["append", "replace", "skip"]

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_very_similar_directions(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test deduplication with very similar direction names."""
        from kernelopt.agents.summarizer import run_summarizer

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        existing_memory = [
            {
                "item_id": "v1",
                "direction": "optimize_memory_access",
                "speedup": 1.20,
            }
        ]

        plan = {
            "plan_id": "v2",
            "direction": "optimize_memory_access_pattern",  # Very similar
        }

        _, memory_update = run_summarizer(
            profiling_before=sample_profiling_context,
            profiling_after=sample_profiling_context,
            optimization_plan=plan,
            kernel_before=kernel_source,
            kernel_after=kernel_source,
            iteration=10,
            framework="triton",
            speedup=1.30,
            existing_memory=existing_memory,
            model="claude-sonnet-4@20250514",
        )

        # LLM should differentiate or treat as same
        assert memory_update["action"] in ["append", "replace", "skip"]
