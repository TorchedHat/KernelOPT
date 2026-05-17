"""Wall-clock benchmarking for end-to-end kernel performance measurement.

Uses triton.testing.do_bench for GPU timing — the same method used by
KernelAgent, KernelSkill, AccelOpt, and the Triton tutorials themselves.

Loads the kernel module, finds the callable + inputs, and benchmarks
inline (no subprocess). This avoids the fragility of subprocess-based
benchmarking (import failures, missing kernel_function, etc.).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _find_callable_and_inputs(module):
    """Find the benchmark-able function and input generator in a kernel module.

    Searches for common patterns across all kernel types:
    - Inductor synthesized: kernel_function + get_inputs
    - Triton tutorials: add/softmax_kernel/matmul + inline test data
    - Helion kernels: kernel_function + get_inputs
    - SOL kernels: run + get_inputs
    """
    fn = (
        getattr(module, "kernel_function", None)
        or getattr(module, "call", None)
        or getattr(module, "run", None)
    )
    get_inputs = getattr(module, "get_inputs", None)
    return fn, get_inputs


def benchmark_kernel(
    kernel_path: str,
    gpu_id: int | None = None,
    warmup: int = 25,
    rep: int = 100,
) -> dict:
    """Benchmark a kernel file end-to-end using triton.testing.do_bench.

    Loads the module inline (same process), finds the callable and inputs,
    and measures wall-clock GPU time. Returns dict with median_ms or error.
    """
    try:
        import torch
        from triton.testing import do_bench
    except ImportError:
        return {"median_ms": -1, "error": "torch or triton not available"}

    if gpu_id is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        mod_name = f"_bench_{Path(kernel_path).stem}_{id(kernel_path)}"
        spec = importlib.util.spec_from_file_location(mod_name, str(kernel_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        return {"median_ms": -1, "error": f"import failed: {str(e)[:200]}"}

    fn, get_inputs = _find_callable_and_inputs(module)

    if fn is None or get_inputs is None:
        return {"median_ms": -1, "error": "no callable or get_inputs found"}

    try:
        inputs = get_inputs()
        with torch.no_grad():
            bench_fn = lambda: fn(*inputs)
            median_ms = do_bench(bench_fn, warmup=warmup, rep=rep)
        return {"median_ms": float(median_ms)}
    except Exception as e:
        return {"median_ms": -1, "error": f"benchmark failed: {str(e)[:200]}"}
