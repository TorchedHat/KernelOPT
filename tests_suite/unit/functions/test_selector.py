"""
Selector tests: pick best PlanCandidate by latency (matches production dataclass).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


def _c(
    plan_id: str,
    *,
    optimization_plan: dict | None = None,
    optimized_kernel: str = "",
    latency_ns: float = float("inf"),
    speedup: float = 1.0,
    profiling_context: dict | None = None,
) -> "PlanCandidate":
    from kernelopt.state import PlanCandidate

    return PlanCandidate(
        plan_id=plan_id,
        optimization_plan=optimization_plan or {},
        optimized_kernel=optimized_kernel,
        latency_ns=latency_ns,
        speedup=speedup,
        profiling_context=profiling_context or {},
    )


class TestSelectBest:
    """Test select_best function."""

    def test_select_best_single_candidate(self):
        from kernelopt.functions.selector import select_best

        candidates = [_c("A", latency_ns=1000, speedup=1.5)]

        best = select_best(candidates)

        assert best is not None
        assert best.plan_id == "A"
        assert best.latency_ns == 1000

    def test_select_best_multiple_candidates(self):
        from kernelopt.functions.selector import select_best

        candidates = [
            _c("A", latency_ns=1000, speedup=1.5),
            _c("B", latency_ns=500, speedup=2.0),
            _c("C", latency_ns=750, speedup=1.7),
        ]

        best = select_best(candidates)

        assert best is not None
        assert best.plan_id == "B"
        assert best.latency_ns == 500

    def test_select_best_empty_list(self):
        from kernelopt.functions.selector import select_best

        assert select_best([]) is None

    def test_select_best_all_invalid(self):
        from kernelopt.functions.selector import select_best

        candidates = [
            _c("A", latency_ns=float("inf"), speedup=0.0),
            _c("B", latency_ns=float("inf"), speedup=0.0),
        ]

        assert select_best(candidates) is None

    def test_select_best_mixed_valid_invalid(self):
        from kernelopt.functions.selector import select_best

        candidates = [
            _c("A", latency_ns=float("inf"), speedup=0.0),
            _c("B", latency_ns=1000, speedup=1.5),
            _c("C", latency_ns=float("inf"), speedup=0.0),
        ]

        best = select_best(candidates)

        assert best is not None
        assert best.plan_id == "B"
        assert best.latency_ns == 1000

    def test_select_best_tie_breaking(self):
        from kernelopt.functions.selector import select_best

        candidates = [
            _c("A", latency_ns=1000, speedup=1.5),
            _c("B", latency_ns=1000, speedup=1.5),
            _c("C", latency_ns=1000, speedup=1.5),
        ]

        best = select_best(candidates)

        assert best is not None
        assert best.plan_id == "A"
        assert best.latency_ns == 1000

    def test_select_best_preserves_all_fields(self):
        from kernelopt.functions.selector import select_best

        candidates = [
            _c(
                "best",
                optimization_plan={"direction": "optimize"},
                optimized_kernel="__global__ void k() {}",
                latency_ns=500,
                speedup=2.0,
                profiling_context={"duration": "500ns"},
            )
        ]

        best = select_best(candidates)

        assert best.plan_id == "best"
        assert best.optimization_plan == {"direction": "optimize"}
        assert best.optimized_kernel == "__global__ void k() {}"
        assert best.latency_ns == 500
        assert best.speedup == 2.0
        assert best.profiling_context == {"duration": "500ns"}

    def test_select_best_ignores_speedup(self):
        from kernelopt.functions.selector import select_best

        candidates = [
            _c("A", latency_ns=1000, speedup=3.0),
            _c("B", latency_ns=500, speedup=1.5),
        ]

        best = select_best(candidates)

        assert best.plan_id == "B"
        assert best.latency_ns == 500

    def test_select_best_very_large_latency(self):
        from kernelopt.functions.selector import select_best

        candidates = [
            _c("A", latency_ns=1e15, speedup=0.001),
            _c("B", latency_ns=1000, speedup=1.5),
        ]

        best = select_best(candidates)

        assert best.plan_id == "B"
        assert best.latency_ns == 1000

    def test_select_best_zero_latency(self):
        from kernelopt.functions.selector import select_best

        candidates = [
            _c("A", latency_ns=0, speedup=float("inf")),
            _c("B", latency_ns=1000, speedup=1.5),
        ]

        best = select_best(candidates)

        assert best.plan_id == "A"
        assert best.latency_ns == 0
