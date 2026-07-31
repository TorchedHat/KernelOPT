"""Install KernelOPT extraction hooks into Inductor's config.

KernelOPT does NOT replace Inductor — it installs read-only hooks
that observe the compilation and extract structural metadata.
The user still calls torch.compile(model, backend="inductor").

Usage:
    from kernelopt.compiler.passes import install_hooks, uninstall_hooks
    install_hooks()
    compiled = torch.compile(model)
    compiled(input)
    uninstall_hooks()
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_SAVED_STATE: dict[str, Any] = {}
_INSTALLED = False


def install_hooks(
    *,
    extract: bool = True,
    triton: bool = True,
) -> None:
    """Install extraction hooks into Inductor config.

    Must be called BEFORE torch.compile(). Hooks are non-destructive
    and do not modify the compilation output.

    Args:
        extract: Install IR/fusion/autotune extraction hooks.
        triton: Install Triton compilation listener.
    """
    global _INSTALLED, _SAVED_STATE

    if _INSTALLED:
        logger.warning("Hooks already installed; call uninstall_hooks() first")
        return

    _SAVED_STATE = {}

    if extract:
        _install_inductor_hooks()

    if triton:
        _install_triton_hooks()

    _INSTALLED = True
    logger.debug("KernelOPT extraction hooks installed")


def uninstall_hooks() -> None:
    """Remove all KernelOPT hooks and restore original config."""
    global _INSTALLED, _SAVED_STATE

    if not _INSTALLED:
        return

    _restore_inductor_hooks()
    _restore_triton_hooks()

    _SAVED_STATE = {}
    _INSTALLED = False
    logger.debug("KernelOPT extraction hooks removed")


def is_installed() -> bool:
    """Check if hooks are currently installed."""
    return _INSTALLED


def _install_inductor_hooks() -> None:
    """Install hooks into torch._inductor.config."""
    try:
        import torch._inductor.config as cfg
    except ImportError:
        logger.warning("torch._inductor not available; skipping Inductor hooks")
        return

    if hasattr(cfg, "pre_grad_custom_pass"):
        _SAVED_STATE["pre_grad_custom_pass"] = cfg.pre_grad_custom_pass
        cfg.pre_grad_custom_pass = _make_pre_grad_pass(cfg.pre_grad_custom_pass)

    if hasattr(cfg, "post_grad_custom_pre_pass"):
        _SAVED_STATE["post_grad_custom_pre_pass"] = cfg.post_grad_custom_pre_pass
        cfg.post_grad_custom_pre_pass = _make_post_grad_pass(cfg.post_grad_custom_pre_pass)


def _install_triton_hooks() -> None:
    """Install Triton compilation listener."""
    try:
        import triton
        if hasattr(triton.knobs.compilation, "listener"):
            _SAVED_STATE["triton_listener"] = triton.knobs.compilation.listener
            triton.knobs.compilation.listener = _triton_listener
    except (ImportError, AttributeError):
        logger.debug("Triton compilation listener not available")


def _restore_inductor_hooks() -> None:
    """Restore original Inductor config values."""
    try:
        import torch._inductor.config as cfg
    except ImportError:
        return

    for key in ("pre_grad_custom_pass", "post_grad_custom_pre_pass"):
        if key in _SAVED_STATE:
            setattr(cfg, key, _SAVED_STATE[key])


def _restore_triton_hooks() -> None:
    """Restore original Triton listener."""
    try:
        import triton
        if "triton_listener" in _SAVED_STATE:
            triton.knobs.compilation.listener = _SAVED_STATE["triton_listener"]
    except (ImportError, AttributeError):
        pass


_captured_pre_grad: list[dict] = []
_captured_post_grad: list[dict] = []
_captured_triton: list[dict] = []


def get_captured_data() -> dict:
    """Return all data captured by installed hooks."""
    return {
        "pre_grad": list(_captured_pre_grad),
        "post_grad": list(_captured_post_grad),
        "triton": list(_captured_triton),
    }


def clear_captured_data() -> None:
    """Clear all captured hook data."""
    _captured_pre_grad.clear()
    _captured_post_grad.clear()
    _captured_triton.clear()


def _make_pre_grad_pass(original: Optional[Callable]) -> Callable:
    """Create a pre-grad pass that captures ops then delegates."""
    def pass_fn(graph):
        try:
            nodes = graph.nodes if hasattr(graph, "nodes") else (
                graph.graph.nodes if hasattr(graph, "graph") else []
            )
            ops = []
            for node in nodes:
                if node.op == "call_function":
                    name = str(node.target).split(".")[-1] if hasattr(node.target, "__name__") else str(node.target)
                    ops.append({"op": name, "args_count": len(node.args)})
            _captured_pre_grad.append({"ops": ops, "num_ops": len(ops)})
        except Exception:
            pass

        if callable(original):
            return original(graph)
        return graph

    return pass_fn


def _make_post_grad_pass(original: Optional[Callable]) -> Callable:
    """Create a post-grad pass that captures IR nodes then delegates."""
    def pass_fn(graph):
        try:
            nodes = graph.nodes if hasattr(graph, "nodes") else (
                graph.graph.nodes if hasattr(graph, "graph") else []
            )
            ir_info = []
            for node in nodes:
                if node.op == "call_function":
                    ir_info.append({
                        "target": str(node.target),
                        "name": node.name,
                    })
            _captured_post_grad.append({"ir_nodes": ir_info, "num_nodes": len(ir_info)})
        except Exception:
            pass

        if callable(original):
            return original(graph)
        return graph

    return pass_fn


def _triton_listener(
    kernel_name: str,
    kernel_src: str,
    metadata: dict,
    **kwargs: Any,
) -> None:
    """Capture Triton compilation events."""
    _captured_triton.append({
        "kernel_name": kernel_name,
        "src_length": len(kernel_src),
        "metadata": {k: str(v)[:100] for k, v in metadata.items()},
    })
