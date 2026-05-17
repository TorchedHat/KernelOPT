"""Wall-clock benchmarking utility for eval_benchmark.py comparisons.

NOT used inside the kernelopt pipeline. The pipeline uses NCU for kernel
performance measurement. This module provides wall-clock timing for
eval_benchmark.py's vs-eager and vs-compile comparison tables.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _find_callable_and_inputs(module):
    """Find the benchmark-able function and input generator in a kernel module."""
    fn = (
        getattr(module, "kernel_function", None)
        or getattr(module, "call", None)
        or getattr(module, "run", None)
    )
    get_inputs = getattr(module, "get_inputs", None)
    return fn, get_inputs


def benchmark_kernel(
    kernel_path: str,
    warmup: int = 25,
    rep: int = 100,
) -> dict:
    """Benchmark a kernel file using triton.testing.do_bench.

    Returns dict with median_ms or error. Used by eval_benchmark.py
    for wall-clock comparison tables, not by the kernelopt pipeline.
    """
    try:
        import torch
        from triton.testing import do_bench
    except ImportError:
        return {"median_ms": -1, "error": "torch or triton not available"}

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
            median_ms = do_bench(lambda: fn(*inputs), warmup=warmup, rep=rep)
        return {"median_ms": float(median_ms)}
    except Exception as e:
        return {"median_ms": -1, "error": f"benchmark failed: {str(e)[:200]}"}
