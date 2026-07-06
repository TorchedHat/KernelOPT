"""Tests that select_best_candidate prefers wall-clock over NCU latency."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "kernelopt"))


@dataclass
class FakeCandidate:
    plan_letter: str
    latency_ns: float
    wall_ms: float
    optimized_kernel: str = "kernel_source"
    correctness_passed: bool = True
    ncu_rep_path: str = ""
    profiling_context: dict = field(default_factory=dict)
    direction: str = "test"
    speedup: float = 1.0
    wall_speedup: float = 1.0
    chain_id: str = ""
    optimization_plan: dict = field(default_factory=lambda: {"confidence": "high"})


def _run_select(candidates, baseline_wall_ms=6.0, baseline_latency_ns=6_000_000):
    from kernelopt.langgraph_pipeline import select_best_candidate

    base_state = {
        "config": MagicMock(T=5, search_strategy="beam", beam_width=1,
                            search_policy="greedy"),
        "kernel_path": "/fake/kernel.py",
        "kernel_source": "baseline",
        "kernel_basename": "kernel",
        "kernel_id": "k0",
        "framework": "triton",
        "output_dir": "/tmp/test_out",
        "ext": ".py",
        "best_kernel": "baseline",
        "best_latency_ns": baseline_latency_ns,
        "best_wall_ms": baseline_wall_ms,
        "best_ncu_rep": "",
        "baseline_latency_ns": baseline_latency_ns,
        "baseline_wall_ms": baseline_wall_ms,
        "baseline_csv": "",
        "current_profiling_context": {},
        "optimization_memory": [],
        "iteration": 1,
        "no_improvement_streak": 0,
        "plans": [],
        "candidates": candidates,
        "iteration_results": [],
        "ncu_config": None,
        "global_best_kernel": "baseline",
        "global_best_latency_ns": baseline_latency_ns,
        "global_best_wall_ms": baseline_wall_ms,
        "active_kernels": [],
        "_entry": "fresh",
    }

    with patch("kernelopt.langgraph_pipeline.get_pipeline_ui", return_value=None), \
         patch("kernelopt.langgraph_pipeline.get_hud_log_console", return_value=None), \
         patch("kernelopt.langgraph_pipeline.console"):
        return select_best_candidate(base_state)


def test_wall_clock_winner_selected_when_all_have_timing():
    """When all candidates have wall-clock data, the wall-clock winner must be selected."""
    candidates = [
        FakeCandidate("A", latency_ns=4_900_000, wall_ms=6.0),   # NCU winner
        FakeCandidate("B", latency_ns=5_500_000, wall_ms=4.8),   # wall-clock winner
        FakeCandidate("C", latency_ns=6_100_000, wall_ms=5.2),
    ]
    result = _run_select(candidates)
    assert result.get("best_wall_ms") == pytest.approx(4.8, abs=0.01), (
        f"Expected best_wall_ms=4.8 (wall-clock winner B), got {result.get('best_wall_ms')}"
    )


def test_ncu_fallback_when_any_wall_missing():
    """When any candidate lacks wall-clock data, fall back to NCU latency."""
    candidates = [
        FakeCandidate("A", latency_ns=4_900_000, wall_ms=float("inf")),  # no wall data
        FakeCandidate("B", latency_ns=5_500_000, wall_ms=4.8),
    ]
    result = _run_select(candidates)
    assert result.get("best_latency_ns") == pytest.approx(4_900_000, rel=0.01), (
        "NCU fallback must select the lowest-latency_ns candidate when wall data is missing"
    )


def test_no_regression_when_all_worse_than_baseline():
    """When all candidates are worse than baseline, best_kernel stays as baseline."""
    candidates = [
        FakeCandidate("A", latency_ns=7_000_000, wall_ms=7.0),
        FakeCandidate("B", latency_ns=8_000_000, wall_ms=8.0),
    ]
    result = _run_select(candidates, baseline_wall_ms=6.0)
    assert result.get("best_kernel") == "baseline", (
        "baseline must be kept when all candidates are worse"
    )


def test_global_best_wall_ms_returned():
    """select_best_candidate must return global_best_wall_ms in its dict."""
    candidates = [
        FakeCandidate("A", latency_ns=5_000_000, wall_ms=4.5),
    ]
    result = _run_select(candidates)
    assert "global_best_wall_ms" in result, (
        "global_best_wall_ms must be present in the return dict to persist across iterations"
    )
