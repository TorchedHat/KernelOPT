"""kernelopt adapter for SOL-ExecBench problems.

Wraps naive Triton kernels into kernelopt-compatible format using
wrap_kernel.py, invokes `kernelopt run`, and collects results.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from benchmarks.sol_execbench.harness import load_problem, resolve_axes
from benchmarks.sol_execbench.wrap_kernel import wrap_kernel


def _detect_kernel_name(source: str) -> Optional[str]:
    match = re.search(r"@triton\.jit\s*\ndef\s+(\w+)", source)
    return match.group(1) if match else None


def _parse_optimization_memory(mem_path: Path) -> list[dict]:
    if not mem_path.exists():
        return []
    items = json.loads(mem_path.read_text())
    return [
        {"direction": it.get("direction", "unknown"),
         "summary": it.get("strategy_description", ""),
         "speedup": it.get("speedup", 0.0)}
        for it in items
    ]


def _parse_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    text = log_path.read_text()
    info = {}
    for line in text.split("\n"):
        if "Baseline:" in line and "ns" in line:
            m = re.search(r"(\d[\d,]*)\s*ns", line.replace(",", ""))
            if m:
                info["baseline_ns"] = float(m.group(1))
        if "Best:" in line and "ns" in line:
            m = re.search(r"(\d[\d,]*)\s*ns", line.replace(",", ""))
            if m:
                info["best_ns"] = float(m.group(1))
    return info


def run_kernelopt_sol(
    naive_kernel_path: str,
    problem_path: str,
    workload_idx: int,
    output_dir: str,
    *,
    T: int = 5,
    N: int = 4,
    K: int = 1,
    gpu_id: int = 0,
    timeout: int = 900,
    replay_mode: str = "kernel",
) -> dict:
    """Run kernelopt on a wrapped SOL-ExecBench kernel for one workload."""
    problem = load_problem(problem_path)
    naive_source = Path(naive_kernel_path).read_text()

    out_dir = Path(output_dir)
    wrapped_path = str(out_dir / f"wrapped_wl{workload_idx}.py")
    wrap_kernel(naive_source, problem, workload_idx, wrapped_path)

    kernel_name = _detect_kernel_name(naive_source)

    cmd = [
        "kernelopt", "run", wrapped_path,
        "-T", str(T), "-N", str(N), "-K", str(K),
        "-o", str(out_dir / f"wl{workload_idx}"), "--no-run-subdir",
        "--replay-mode", replay_mode,
    ]
    if kernel_name:
        cmd.extend(["--kernel-name", kernel_name])
    cmd.extend(["--kernel-args", f"--profile-only --workload-idx {workload_idx}"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        wall_clock = time.monotonic() - start
        success = proc.returncode == 0
    except subprocess.TimeoutExpired:
        wall_clock = timeout
        success = False

    wl_out = out_dir / f"wl{workload_idx}"
    result = {
        "tool": "kernelopt",
        "workload_idx": workload_idx,
        "axes": resolve_axes(problem, workload_idx),
        "success": success,
        "wall_clock_s": round(wall_clock, 1),
    }

    best = wl_out / "best_kernel.py"
    if best.exists():
        result["best_kernel_path"] = str(best)

    result["iterations"] = _parse_optimization_memory(wl_out / "optimization_memory.json")
    result.update(_parse_log(wl_out / "optimization_log.md"))

    return result


def run_kernelopt_track_a(
    naive_kernel_path: str,
    problem_path: str,
    problem_id: str,
    results_dir: str,
    *,
    T: int = 5,
    N: int = 4,
    K: int = 1,
    gpu_id: int = 0,
    workload_indices: Optional[list[int]] = None,
) -> dict:
    """Track A: optimize a naive kernel across all (or selected) workloads."""
    problem = load_problem(problem_path)
    if workload_indices is None:
        workload_indices = list(range(problem.num_workloads))

    out_dir = Path(results_dir) / "kernelopt" / "track_a" / problem_id
    workload_results = []

    for wi in workload_indices:
        r = run_kernelopt_sol(
            naive_kernel_path, problem_path, wi, str(out_dir),
            T=T, N=N, K=K, gpu_id=gpu_id,
        )
        workload_results.append(r)

    return {
        "problem_id": problem_id,
        "problem_name": problem.name,
        "category": problem.category,
        "tool": "kernelopt",
        "track": "A",
        "workloads": workload_results,
        "total_wall_clock_s": sum(r["wall_clock_s"] for r in workload_results),
    }
