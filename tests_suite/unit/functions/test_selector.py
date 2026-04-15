"""
Selector Tests.

Tests for candidate selection logic.
Tests selecting the best candidate from a list based on latency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestSelectBest:
    """Test select_best function."""

    def test_select_best_single_candidate(self):
        """Test selecting from single candidate."""
        from kernelopt.functions.selector import select_best
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(
                plan_id="A",
                plan={},
                optimized_source="",
                latency_ns=1000,
                speedup=1.5,
            )
        ]

        best = select_best(candidates)

        assert best is not None
        assert best.plan_id == "A"
        assert best.latency_ns == 1000

    def test_select_best_multiple_candidates(self):
        """Test selecting best from multiple candidates."""
        from kernelopt.functions.selector import select_best
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(plan_id="A", plan={}, optimized_source="", latency_ns=1000, speedup=1.5),
            PlanCandidate(plan_id="B", plan={}, optimized_source="", latency_ns=500, speedup=2.0),
            PlanCandidate(plan_id="C", plan={}, optimized_source="", latency_ns=750, speedup=1.7),
        ]

        best = select_best(candidates)

        assert best is not None
        assert best.plan_id == "B"  # Lowest latency
        assert best.latency_ns == 500

    def test_select_best_empty_list(self):
        """Test selecting from empty candidate list."""
        from kernelopt.functions.selector import select_best

        candidates = []

        best = select_best(candidates)

        assert best is None

    def test_select_best_all_invalid(self):
        """Test selecting when all candidates have infinite latency."""
        from kernelopt.functions.selector import select_best
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(plan_id="A", plan={}, optimized_source="", latency_ns=float("inf"), speedup=0.0),
            PlanCandidate(plan_id="B", plan={}, optimized_source="", latency_ns=float("inf"), speedup=0.0),
        ]

        best = select_best(candidates)

        assert best is None

    def test_select_best_mixed_valid_invalid(self):
        """Test selecting with mix of valid and invalid candidates."""
        from kernelopt.functions.selector import select_best
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(plan_id="A", plan={}, optimized_source="", latency_ns=float("inf"), speedup=0.0),
            PlanCandidate(plan_id="B", plan={}, optimized_source="", latency_ns=1000, speedup=1.5),
            PlanCandidate(plan_id="C", plan={}, optimized_source="", latency_ns=float("inf"), speedup=0.0),
        ]

        best = select_best(candidates)

        assert best is not None
        assert best.plan_id == "B"  # Only valid candidate
        assert best.latency_ns == 1000

    def test_select_best_tie_breaking(self):
        """Test tie-breaking when candidates have same latency."""
        from kernelopt.functions.selector import select_best
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(plan_id="A", plan={}, optimized_source="", latency_ns=1000, speedup=1.5),
            PlanCandidate(plan_id="B", plan={}, optimized_source="", latency_ns=1000, speedup=1.5),
            PlanCandidate(plan_id="C", plan={}, optimized_source="", latency_ns=1000, speedup=1.5),
        ]

        best = select_best(candidates)

        assert best is not None
        # When tied, min() returns first occurrence
        assert best.plan_id == "A"
        assert best.latency_ns == 1000

    def test_select_best_preserves_all_fields(self):
        """Test that selected candidate preserves all fields."""
        from kernelopt.functions.selector import select_best
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(
                plan_id="best",
                plan={"direction": "optimize"},
                optimized_source="__global__ void k() {}",
                latency_ns=500,
                speedup=2.0,
                profiling_ctx={"duration": "500ns"},
                ncu_config={"kernel_name": "k"},
                framework="cuda",
            )
        ]

        best = select_best(candidates)

        assert best.plan_id == "best"
        assert best.plan == {"direction": "optimize"}
        assert best.optimized_source == "__global__ void k() {}"
        assert best.latency_ns == 500
        assert best.speedup == 2.0
        assert best.profiling_ctx == {"duration": "500ns"}

    def test_select_best_ignores_speedup(self):
        """Test that selection is based on latency, not speedup."""
        from kernelopt.functions.selector import select_best
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(plan_id="A", plan={}, optimized_source="", latency_ns=1000, speedup=3.0),  # High speedup
            PlanCandidate(plan_id="B", plan={}, optimized_source="", latency_ns=500, speedup=1.5),   # Low speedup but better latency
        ]

        best = select_best(candidates)

        # Should select based on latency, not speedup
        assert best.plan_id == "B"
        assert best.latency_ns == 500

    def test_select_best_very_large_latency(self):
        """Test handling very large latency values."""
        from kernelopt.functions.selector import select_best
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(plan_id="A", plan={}, optimized_source="", latency_ns=1e15, speedup=0.001),
            PlanCandidate(plan_id="B", plan={}, optimized_source="", latency_ns=1000, speedup=1.5),
        ]

        best = select_best(candidates)

        assert best.plan_id == "B"
        assert best.latency_ns == 1000

    def test_select_best_zero_latency(self):
        """Test handling zero latency (edge case)."""
        from kernelopt.functions.selector import select_best
        from kernelopt.state import PlanCandidate

        candidates = [
            PlanCandidate(plan_id="A", plan={}, optimized_source="", latency_ns=0, speedup=float("inf")),
            PlanCandidate(plan_id="B", plan={}, optimized_source="", latency_ns=1000, speedup=1.5),
        ]

        best = select_best(candidates)

        # Zero latency is lowest
        assert best.plan_id == "A"
        assert best.latency_ns == 0
