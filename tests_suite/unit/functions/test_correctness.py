"""
Unit tests for correctness checking functions.

Tests cover:
- Correctness validation subprocess execution
- Kernel function extraction
- Framework-specific correctness checks
- Error handling
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestCorrectness:
    """Test suite for kernel correctness checking."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_check_correctness_subprocess_valid(self, test_kernel_path, skip_if_no_cuda):
        """Test correctness check with valid kernel."""
        from kernelopt.functions.correctness import check_correctness_subprocess

        kernel_source = test_kernel_path.read_text()
        passed, msg = check_correctness_subprocess(
            str(test_kernel_path), kernel_source, "triton"
        )

        # Same kernel should pass correctness
        assert passed is True
        assert msg != ""

    @pytest.mark.skip(reason="_extract_kernel_function is private/removed - test public API only")
    def test_extract_kernel_function_triton(self):
        """Test extracting kernel function from Triton code."""
        pass

    @pytest.mark.skip(reason="_extract_kernel_function is private/removed - test public API only")
    def test_extract_kernel_function_cuda(self):
        """Test extracting kernel function from CUDA code."""
        pass

    def test_correctness_check_invalid_framework(self, test_kernel_path):
        """Test correctness check with invalid framework."""
        from kernelopt.functions.correctness import check_correctness_subprocess

        kernel_source = test_kernel_path.read_text()

        # Invalid framework should be handled
        passed, msg = check_correctness_subprocess(
            str(test_kernel_path), kernel_source, "invalid_framework"
        )

        # Should fail or handle gracefully
        assert isinstance(passed, bool)
        assert isinstance(msg, str)


class TestSelector:
    """Test suite for plan selection logic."""

    def test_select_best_candidate_by_latency(self):
        """Test selecting best candidate by latency."""
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(
                plan_letter="A",
                direction="opt_a",
                latency_ns=100.0,
                speedup=1.5,
            ),
            PlanCandidate(
                plan_letter="B",
                direction="opt_b",
                latency_ns=80.0,
                speedup=1.8,
            ),
            PlanCandidate(
                plan_letter="C",
                direction="opt_c",
                latency_ns=120.0,
                speedup=1.2,
            ),
        ]

        best = min(candidates, key=lambda c: c.latency_ns)
        assert best.plan_letter == "B"
        assert best.latency_ns == 80.0

    def test_select_best_candidate_correctness(self):
        """Test that only correct candidates are considered."""
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(
                plan_letter="A",
                direction="opt_a",
                latency_ns=80.0,
                correctness_passed=False,
            ),
            PlanCandidate(
                plan_letter="B",
                direction="opt_b",
                latency_ns=100.0,
                correctness_passed=True,
            ),
        ]

        valid = [c for c in candidates if c.correctness_passed]
        assert len(valid) == 1
        assert valid[0].plan_letter == "B"

    def test_select_candidates_by_speedup(self):
        """Test selecting candidates sorted by speedup."""
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(plan_letter="A", speedup=1.2, correctness_passed=True),
            PlanCandidate(plan_letter="B", speedup=1.8, correctness_passed=True),
            PlanCandidate(plan_letter="C", speedup=1.1, correctness_passed=True),
        ]

        sorted_candidates = sorted(
            [c for c in candidates if c.correctness_passed],
            key=lambda c: c.speedup,
            reverse=True
        )

        assert sorted_candidates[0].plan_letter == "B"
        assert sorted_candidates[0].speedup == 1.8
