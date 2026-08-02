"""Bridge for injecting KernelOPT-optimized kernels into JAX via Pallas.

Pallas provides the path from KernelOPT's optimized Triton kernels
into JAX's execution pipeline:
  1. KernelOPT generates an optimized Triton kernel
  2. The kernel is wrapped as a Pallas kernel function
  3. pl.pallas_call() lowers it to Triton IR
  4. XLA injects it via stablehlo.custom_call("__gpu$xla.gpu.triton")

References:
    - Pallas Triton registration: jax/_src/pallas/triton/pallas_call_registration.py
    - Pallas lowering: jax/_src/pallas/triton/lowering.py
    - CompilerParams: jax/_src/pallas/triton/core.py
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from kernelopt.compiler.version_compat import detect_capabilities

logger = logging.getLogger(__name__)

_capabilities: dict | None = None


def _get_caps() -> dict:
    """Return cached capability dict (computed once per process)."""
    global _capabilities
    if _capabilities is None:
        _capabilities = detect_capabilities()
    return _capabilities


def create_pallas_kernel_wrapper(
    kernel_fn: Callable,
    out_shape: Any,
    grid: tuple[int, ...],
    num_warps: int = 4,
    num_stages: int = 1,
) -> Callable:
    """Wrap a Triton-style kernel function for use with Pallas.

    Args:
        kernel_fn: Function using Pallas primitives (pl.load, pl.store, etc.)
        out_shape: jax.ShapeDtypeStruct describing the output
        grid: Launch grid dimensions
        num_warps: Number of warps per block
        num_stages: Pipeline stages

    Returns:
        A callable that takes JAX arrays and returns optimized output.

    Raises:
        RuntimeError: If JAX Pallas or its Triton backend is not installed.
    """
    caps = _get_caps()

    if not caps.get("has_pallas"):
        raise RuntimeError(
            "Pallas bridge requires jax.experimental.pallas. "
            "Install with: pip install jax[cuda12]"
        )

    if not caps.get("has_pallas_triton", caps.get("has_pallas")):
        raise RuntimeError(
            "Pallas Triton backend not available. "
            "Requires jax.experimental.pallas.triton."
        )

    from jax.experimental import pallas as pl

    try:
        from jax.experimental.pallas.triton import CompilerParams as TritonCompilerParams
    except ImportError:
        from jax.experimental.pallas import triton as _plt
        TritonCompilerParams = _plt.CompilerParams

    def wrapped(*args):
        return pl.pallas_call(
            kernel_fn,
            out_shape=out_shape,
            grid=grid,
            compiler_params=TritonCompilerParams(
                num_warps=num_warps,
                num_stages=num_stages,
            ),
        )(*args)

    return wrapped


def check_pallas_available() -> bool:
    """Check if JAX Pallas with Triton backend is available."""
    caps = _get_caps()
    return bool(caps.get("has_pallas"))
