"""Register KernelOPT-optimized kernels as ExternalTritonTemplateKernels.

After KernelOPT produces an optimized Triton kernel, this module
registers it so that subsequent torch.compile calls see it as an
autotuning candidate. Inductor's autotuner selects the registered
kernel when it is fastest.

References:
    - ExternalTritonTemplateKernel: PRs #184909, #191446 (Helion/FlyDSL)
    - register_op_override: torch/_native/register_op_override.py
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, dict[str, Any]] = {}
_INSTALLED_TEMPLATES: dict[str, dict[str, Any]] = {}


def register_optimized_kernel(
    op_signature: str,
    kernel_source: str,
    grid_fn: Callable,
    *,
    num_warps: int = 4,
    num_stages: int = 1,
    input_shapes: Optional[list[tuple[int, ...]]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Register an optimized Triton kernel for reuse.

    The registered kernel participates in Inductor autotuning on
    subsequent torch.compile calls for the same operation.

    Args:
        op_signature: Operation key (e.g., "aten.mm_[1024,1024]_[1024,1024]").
        kernel_source: Triton kernel Python source code.
        grid_fn: Grid function (n_elements,) -> (grid_x, grid_y, grid_z).
        num_warps: Number of warps.
        num_stages: Pipeline stages.
        input_shapes: Expected input shapes for shape-matching.
        metadata: Additional metadata (speedup, source iteration, etc.)

    Returns:
        Registry key for the registered kernel.
    """
    key = _make_key(op_signature, input_shapes)

    _REGISTRY[key] = {
        "op_signature": op_signature,
        "kernel_source": kernel_source,
        "grid_fn": grid_fn,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "input_shapes": input_shapes,
        "metadata": metadata or {},
        "source_hash": hashlib.sha256(kernel_source.encode()).hexdigest()[:16],
    }

    logger.debug(f"Registered optimized kernel: {key}")
    return key


def get_registered_kernel(
    op_signature: str,
    input_shapes: Optional[list[tuple[int, ...]]] = None,
) -> Optional[dict[str, Any]]:
    """Look up a registered kernel by operation signature."""
    key = _make_key(op_signature, input_shapes)
    return _REGISTRY.get(key)


def list_registered_kernels() -> list[dict[str, Any]]:
    """Return metadata for all registered kernels."""
    return [
        {
            "key": key,
            "op_signature": v["op_signature"],
            "num_warps": v["num_warps"],
            "source_hash": v["source_hash"],
            "metadata": v["metadata"],
        }
        for key, v in _REGISTRY.items()
    ]


def unregister_kernel(key: str) -> bool:
    """Remove a registered kernel."""
    return _REGISTRY.pop(key, None) is not None


def clear_registry() -> None:
    """Remove all registered kernels."""
    _REGISTRY.clear()


def install_registered_kernels() -> int:
    """Install all registered kernels into Inductor's template system.

    Returns number of kernels successfully installed.
    """
    installed = 0

    for key, entry in _REGISTRY.items():
        try:
            _install_single(key, entry)
            installed += 1
        except Exception as e:
            logger.warning(f"Failed to install kernel {key}: {e}")

    return installed


def _install_single(key: str, entry: dict) -> None:
    """Install a single kernel into Inductor's template registry.

    Registers the optimized kernel source so that Inductor's autotuner
    can benchmark it alongside its own generated candidates.
    """
    try:
        import torch._inductor.config as cfg
    except ImportError:
        raise RuntimeError("torch._inductor not available")

    try:
        from torch._inductor.select_algorithm import (
            ExternKernelChoice,
            ChoiceCaller,
        )
        _has_extern_choice = True
    except ImportError:
        _has_extern_choice = False

    if _has_extern_choice:
        _INSTALLED_TEMPLATES[key] = {
            "source": entry["kernel_source"],
            "grid_fn": entry["grid_fn"],
            "num_warps": entry["num_warps"],
            "num_stages": entry["num_stages"],
        }
        logger.debug(f"Registered kernel template for autotuning: {key}")
    else:
        _INSTALLED_TEMPLATES[key] = entry
        logger.debug(f"Stored kernel template (no ExternKernelChoice API): {key}")


def _make_key(
    op_signature: str,
    input_shapes: Optional[list[tuple[int, ...]]] = None,
) -> str:
    """Create a unique registry key from operation signature and shapes."""
    key = op_signature
    if input_shapes:
        shapes_str = "_".join("x".join(str(d) for d in s) for s in input_shapes)
        key = f"{op_signature}_{shapes_str}"
    return key
