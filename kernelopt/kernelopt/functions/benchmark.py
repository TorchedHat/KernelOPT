"""Wall-clock benchmarking for end-to-end kernel performance measurement.

Measures the full computation (all GPU kernels including library calls)
using CUDA Events timing. This is the primary speedup metric — NCU
remains for per-kernel hardware diagnosis.

Design:
  - Runs the kernel file as a subprocess to avoid polluting the parent's
    CUDA context (important for EXCLUSIVE_PROCESS GPU mode).
  - Uses CUDA Events (not CPU timers) for accurate GPU timing.
  - Reports median of 100 runs after 25 warmup iterations.
  - Works for any framework: Triton, CUDA, PyTorch.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WARMUP_REPS = 25
TIMING_REPS = 100


def benchmark_kernel(
    kernel_path: str,
    gpu_id: int | None = None,
    warmup: int = WARMUP_REPS,
    reps: int = TIMING_REPS,
    timeout: int = 120,
) -> dict:
    """Benchmark a kernel file end-to-end using CUDA Events in a subprocess.

    Returns dict with keys: median_ms, min_ms, max_ms, error.
    The subprocess isolation ensures we don't hold a CUDA context in the
    parent process (critical for EXCLUSIVE_PROCESS GPU mode and cascade
    prevention).
    """
    script = f"""\
import json, sys, os, torch

torch.cuda.set_device(0)
kernel_path = sys.argv[1]

import importlib.util
spec = importlib.util.spec_from_file_location("_bench_mod", kernel_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["_bench_mod"] = mod
spec.loader.exec_module(mod)

fn = getattr(mod, "kernel_function", None) or getattr(mod, "call", None)
get_inputs = getattr(mod, "get_inputs", None)

if fn is None or get_inputs is None:
    print(json.dumps({{"error": "kernel_function or get_inputs not found", "median_ms": -1}}))
    sys.exit(0)

try:
    inputs = get_inputs()
    with torch.no_grad():
        for _ in range({warmup}):
            fn(*inputs)
        torch.cuda.synchronize()

        times = []
        for _ in range({reps}):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        median_ms = times[len(times) // 2]
        print(json.dumps({{"median_ms": median_ms, "min_ms": times[0], "max_ms": times[-1]}}))
except Exception as e:
    print(json.dumps({{"error": str(e)[:300], "median_ms": -1}}))
"""

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, str(kernel_path)],
            capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    return json.loads(line)
        return {
            "median_ms": -1,
            "error": f"exit={proc.returncode}: {(proc.stderr or '')[-200:]}",
        }
    except subprocess.TimeoutExpired:
        return {"median_ms": -1, "error": f"benchmark timeout ({timeout}s)"}
    except Exception as e:
        return {"median_ms": -1, "error": str(e)[:200]}
