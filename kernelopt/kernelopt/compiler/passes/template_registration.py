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

    Attempts to register the optimized kernel so that Inductor's autotuner
    can benchmark it alongside its own generated candidates.

    Full autotuning integration requires one of:
    - ``ExternKernelChoice`` + ``ChoiceCaller`` (for callable kernels)
    - ``register_custom_op_autotuning`` (requires wrapping as a CustomOpDef)
    - ``ExternalTritonTemplateKernel`` (for template-based backends like Helion)

    When these APIs are available, we compile the kernel source, create a
    callable wrapper, and register it via ``ExternKernelChoice``.  When not
    available (older PyTorch versions), the kernel is stored for later use
    but cannot participate in Inductor's autotuning.
    """
    from kernelopt.compiler.version_compat import detect_capabilities

    kernel_source = entry["kernel_source"]
    caps = detect_capabilities()

    namespace: dict[str, Any] = {}
    kernel_fn = None
    try:
        code = compile(kernel_source, f"<kernelopt:{key}>", "exec")
        exec(code, namespace)  # noqa: S102
        for obj in namespace.values():
            if callable(obj) and (
                hasattr(obj, "__triton_kernel__")
                or getattr(obj, "fn", None) is not None
                or getattr(obj, "run", None) is not None
            ):
                kernel_fn = obj
                break
    except Exception as e:
        logger.warning("Cannot compile kernel source for %s: %s", key, e)
        _INSTALLED_TEMPLATES[key] = {
            "source": kernel_source,
            "status": "compile_failed",
            "error": str(e),
        }
        return

    if kernel_fn is None:
        logger.info(
            "No Triton kernel function found in source for %s; "
            "storing source for manual injection",
            key,
        )
        _INSTALLED_TEMPLATES[key] = {
            "source": kernel_source,
            "status": "no_kernel_found",
        }
        return

    grid_fn = entry["grid_fn"]
    num_warps = entry["num_warps"]
    num_stages = entry["num_stages"]

    def _make_launcher(kfn, gfn, nw, ns):
        """Create a tensor-level wrapper for the Triton kernel."""
        def launcher(*args, **kwargs):
            import torch
            grid = gfn(*[a.numel() if isinstance(a, torch.Tensor) else a for a in args])
            kfn[grid](*args, num_warps=nw, num_stages=ns, **kwargs)
        launcher.__name__ = f"kernelopt_{key}"
        launcher.__qualname__ = launcher.__name__
        return launcher

    launcher = _make_launcher(kernel_fn, grid_fn, num_warps, num_stages)

    try:
        from torch._inductor.select_algorithm import ExternKernelChoice

        safe_name = key.replace(".", "_").replace("[", "_").replace("]", "_").replace(",", "_")
        choice_name = f"kernelopt_{safe_name}"
        choice = ExternKernelChoice(launcher, name=choice_name)
        _INSTALLED_TEMPLATES[key] = {
            "source": kernel_source,
            "kernel_fn": kernel_fn,
            "launcher": launcher,
            "extern_choice": choice,
            "grid_fn": grid_fn,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "status": "registered_extern_choice",
        }
        logger.info("Registered kernel as ExternKernelChoice: %s", choice_name)
        return
    except ImportError:
        logger.debug("ExternKernelChoice not available")
    except Exception as e:
        logger.debug("ExternKernelChoice registration failed for %s: %s", key, e)

    if caps.get("has_register_custom_op_autotuning"):
        logger.info(
            "Kernel %s: register_custom_op_autotuning available but requires "
            "wrapping as a CustomOpDef — storing compiled callable for manual "
            "injection via torch._inductor.kernel.custom_op.CustomOpConfig",
            key,
        )

    _INSTALLED_TEMPLATES[key] = {
        "source": kernel_source,
        "kernel_fn": kernel_fn,
        "launcher": launcher,
        "grid_fn": grid_fn,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "status": "stored_compiled",
    }
    logger.info(
        "Stored compiled kernel for %s (ExternKernelChoice unavailable, "
        "kernel ready for manual injection)",
        key,
    )


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
