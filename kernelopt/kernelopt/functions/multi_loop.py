"""Outer optimization loop for multi-kernel top-K runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from kernelopt.schemas import KernelTarget


@dataclass
class MultiKernelResult:
    target: KernelTarget
    output_dir: Path
    speedup: float
    baseline_ns: float
    best_ns: float
    failed: bool
    error: str = ""


def run_multi_kernel_loop(
    targets: list[KernelTarget],
    k: int,
    run_dir: Path,
    pipeline_fn: Callable[[KernelTarget, Path], MultiKernelResult],
    reprofiling_threshold: float,
    re_rank_fn: Callable[[list[KernelTarget]], list[KernelTarget]],
) -> list[MultiKernelResult]:
    """Run the existing single-kernel pipeline for each of the top-K targets.

    Args:
        targets: Ordered list of custom KernelTargets (hottest first).
        k: Maximum number of kernels to optimize.
        run_dir: Root run directory; per-kernel subdirs created here.
        pipeline_fn: Callable(target, output_dir) → MultiKernelResult.
        reprofiling_threshold: Fractional speedup above which re-ranking is triggered.
        re_rank_fn: Callable(remaining) → reordered remaining list.
    """
    remaining = list(targets)
    results: list[MultiKernelResult] = []

    while remaining and len(results) < k:
        target = remaining.pop(0)
        per_kernel_dir = run_dir / f"kernel_{len(results) + 1}_{target.kernel_name}"
        per_kernel_dir.mkdir(parents=True, exist_ok=True)

        result = pipeline_fn(target, per_kernel_dir)
        results.append(result)

        if not result.failed and (result.speedup - 1.0) > reprofiling_threshold:
            remaining = re_rank_fn(remaining)

    return results
