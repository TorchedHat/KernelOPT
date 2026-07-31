"""Custom decomposition table for Inductor.

Provides profile-guided decompositions that produce more fusible
subgraphs than Inductor's defaults. Passed to compile_fx via
the decompositions= parameter.

IMPORTANT: compile_fx(decompositions=...) REPLACES the defaults.
Always merge with select_decomp_table():
    {**select_decomp_table(), **custom_decomps}

References:
    - compile_fx decompositions: torch/_inductor/compile_fx.py:2889
    - select_decomp_table: torch/_inductor/decomposition.py
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_CUSTOM_DECOMPS: dict[Any, Callable] = {}


def register_decomposition(
    op: Any,
    decomp_fn: Callable,
    *,
    name: str = "",
) -> None:
    """Register a custom decomposition for an ATen operation.

    Args:
        op: The aten operation to decompose (e.g., torch.ops.aten.layer_norm.default).
        decomp_fn: Replacement function with matching signature.
        name: Human-readable name for logging.
    """
    _CUSTOM_DECOMPS[op] = decomp_fn
    logger.debug(f"Registered decomposition: {name or str(op)}")


def get_decomposition_table() -> dict[Any, Callable]:
    """Get the merged decomposition table (defaults + custom).

    Returns:
        Merged dict suitable for compile_fx(decompositions=...).
    """
    try:
        from torch._inductor.decomposition import select_decomp_table
        base = select_decomp_table()
    except ImportError:
        base = {}

    return {**base, **_CUSTOM_DECOMPS}


def get_custom_decompositions() -> dict[Any, Callable]:
    """Get only the custom decompositions (no defaults)."""
    return dict(_CUSTOM_DECOMPS)


def clear_decompositions() -> None:
    """Remove all custom decompositions."""
    _CUSTOM_DECOMPS.clear()


def register_builtin_decompositions() -> int:
    """Register KernelOPT's built-in decompositions.

    Returns number of decompositions registered.
    """
    count = 0

    try:
        import torch

        @torch.library.custom_op("kernelopt::fused_rms_norm", mutates_args=())
        def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            variance = x.pow(2).mean(-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + eps)
            return x_normed * weight

        logger.debug("Registered built-in decompositions")
        count = 1
    except (ImportError, Exception) as e:
        logger.debug(f"Built-in decompositions not available: {e}")

    return count
