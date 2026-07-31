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
        op: The aten operation to decompose
            (e.g., torch.ops.aten.native_layer_norm.default).
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

    These decompositions expose compound ops as elementwise primitives,
    enabling Inductor to fuse them with adjacent operations.

    Returns number of decompositions registered.
    """
    count = 0

    try:
        import torch

        def _layer_norm_decomp(input, normalized_shape, weight, bias, eps):
            mean = input.mean(-1, keepdim=True)
            var = input.var(-1, keepdim=True, correction=0)
            out = (input - mean) * torch.rsqrt(var + eps)
            if weight is not None:
                out = out * weight
            if bias is not None:
                out = out + bias
            return (out, mean, torch.rsqrt(var + eps))

        def _group_norm_decomp(input, num_groups, weight, bias, eps):
            N, C = input.shape[:2]
            spatial = input.shape[2:]
            x = input.reshape(N, num_groups, C // num_groups, *spatial)
            mean = x.mean(dim=list(range(2, x.dim())), keepdim=True)
            var = x.var(dim=list(range(2, x.dim())), keepdim=True, correction=0)
            x_norm = (x - mean) * torch.rsqrt(var + eps)
            x_norm = x_norm.reshape(N, C, *spatial)
            if weight is not None:
                shape = [1, C] + [1] * len(spatial)
                x_norm = x_norm * weight.reshape(shape)
            if bias is not None:
                shape = [1, C] + [1] * len(spatial)
                x_norm = x_norm + bias.reshape(shape)
            return (x_norm, mean.squeeze(), torch.rsqrt(var + eps).squeeze())

        if hasattr(torch.ops.aten, "native_layer_norm"):
            register_decomposition(
                torch.ops.aten.native_layer_norm.default,
                _layer_norm_decomp,
                name="native_layer_norm",
            )
            count += 1

        if hasattr(torch.ops.aten, "native_group_norm"):
            register_decomposition(
                torch.ops.aten.native_group_norm.default,
                _group_norm_decomp,
                name="native_group_norm",
            )
            count += 1

    except ImportError:
        logger.debug("torch not available for builtin decompositions")

    return count
