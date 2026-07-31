"""Install KernelOPT extraction hooks into Inductor's config.

KernelOPT does NOT replace Inductor — it installs read-only hooks
that observe the compilation and extract structural metadata.
The user still calls torch.compile(model, backend="inductor").

Hooks installed by install_hooks(extract=True):
  - pre_grad_custom_pass: pre-decomposition FX graph capture
  - post_grad_custom_pre_pass: post-decomposition IR node capture
  - _pre_fusion_custom_pass: pre-fusion scheduler node capture
  - _post_fusion_custom_pass: post-fusion scheduler node capture
  - WhyNoFuse patch: fusion rejection reasons

Hooks installed by install_hooks(triton=True):
  - triton.knobs.compilation.listener: Triton compilation metadata
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_SAVED_STATE: dict[str, Any] = {}
_INSTALLED = False

_captured_pre_grad: list[dict] = []
_captured_post_grad: list[dict] = []
_captured_fusion_pre: list[dict] = []
_captured_fusion_post: list[dict] = []
_captured_fusion_rejections: list[dict] = []
_captured_triton: list[dict] = []


def install_hooks(
    *,
    extract: bool = True,
    triton: bool = True,
) -> None:
    """Install extraction hooks into Inductor config.

    Must be called BEFORE torch.compile(). Hooks are non-destructive
    and do not modify the compilation output.

    Args:
        extract: Install all Inductor extraction hooks (pre-grad, post-grad,
                 pre/post-fusion, WhyNoFuse).
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


def get_captured_data() -> dict:
    """Return all data captured by installed hooks."""
    return {
        "pre_grad": list(_captured_pre_grad),
        "post_grad": list(_captured_post_grad),
        "fusion_pre": list(_captured_fusion_pre),
        "fusion_post": list(_captured_fusion_post),
        "fusion_rejections": list(_captured_fusion_rejections),
        "triton": list(_captured_triton),
    }


def clear_captured_data() -> None:
    """Clear all captured hook data."""
    _captured_pre_grad.clear()
    _captured_post_grad.clear()
    _captured_fusion_pre.clear()
    _captured_fusion_post.clear()
    _captured_fusion_rejections.clear()
    _captured_triton.clear()


def _install_inductor_hooks() -> None:
    """Install all extraction hooks into torch._inductor.config."""
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

    if hasattr(cfg, "_pre_fusion_custom_pass"):
        _SAVED_STATE["_pre_fusion_custom_pass"] = cfg._pre_fusion_custom_pass
        cfg._pre_fusion_custom_pass = _make_fusion_pass(
            cfg._pre_fusion_custom_pass, _captured_fusion_pre
        )

    if hasattr(cfg, "_post_fusion_custom_pass"):
        _SAVED_STATE["_post_fusion_custom_pass"] = cfg._post_fusion_custom_pass
        cfg._post_fusion_custom_pass = _make_fusion_pass(
            cfg._post_fusion_custom_pass, _captured_fusion_post
        )

    _patch_why_no_fuse()


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

    hook_keys = (
        "pre_grad_custom_pass",
        "post_grad_custom_pre_pass",
        "_pre_fusion_custom_pass",
        "_post_fusion_custom_pass",
    )
    for key in hook_keys:
        if key in _SAVED_STATE:
            setattr(cfg, key, _SAVED_STATE[key])

    _unpatch_why_no_fuse()


def _restore_triton_hooks() -> None:
    """Restore original Triton listener."""
    try:
        import triton
        if "triton_listener" in _SAVED_STATE:
            triton.knobs.compilation.listener = _SAVED_STATE["triton_listener"]
    except (ImportError, AttributeError):
        pass


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


def _make_fusion_pass(
    original: Optional[Callable],
    capture_list: list[dict],
) -> Callable:
    """Create a fusion pass that captures scheduler nodes then delegates.

    Pass-through: returns nodes unchanged (read-only capture).
    """
    def pass_fn(nodes):
        try:
            node_info = []
            for node in nodes:
                info = {"name": getattr(node, "name", str(node))}
                if hasattr(node, "get_nodes"):
                    info["sub_nodes"] = len(list(node.get_nodes()))
                node_info.append(info)
            capture_list.append({"nodes": node_info, "count": len(node_info)})
        except Exception:
            pass

        if callable(original):
            return original(nodes)
        return nodes

    return pass_fn


def _patch_why_no_fuse() -> None:
    """Wrap WhyNoFuse.__call__ to capture fusion rejection reasons."""
    try:
        from torch._inductor.scheduler import WhyNoFuse
    except ImportError:
        return

    if hasattr(WhyNoFuse, "_kernelopt_original_call"):
        return

    original_call = WhyNoFuse.__call__

    def patched_call(self, node1, node2):
        result = original_call(self, node1, node2)
        try:
            if self.args:
                _captured_fusion_rejections.append({
                    "node1": getattr(node1, "name", str(node1)),
                    "node2": getattr(node2, "name", str(node2)),
                    "reason": str(self.args[-1])[:200] if self.args else "",
                })
        except Exception:
            pass
        return result

    WhyNoFuse._kernelopt_original_call = original_call
    WhyNoFuse.__call__ = patched_call
    _SAVED_STATE["_why_no_fuse_patched"] = True


def _unpatch_why_no_fuse() -> None:
    """Restore original WhyNoFuse.__call__."""
    if "_why_no_fuse_patched" not in _SAVED_STATE:
        return
    try:
        from torch._inductor.scheduler import WhyNoFuse
        if hasattr(WhyNoFuse, "_kernelopt_original_call"):
            WhyNoFuse.__call__ = WhyNoFuse._kernelopt_original_call
            del WhyNoFuse._kernelopt_original_call
    except ImportError:
        pass


def _triton_listener(
    *,
    src: str,
    metadata: Any,
    metadata_group: Any = None,
    times: Any = None,
    cache_hit: bool = False,
) -> None:
    """Capture Triton compilation events.

    Matches the CompilationListener protocol:
      triton.knobs.compilation.listener(*, src, metadata, metadata_group, times, cache_hit)
    """
    if cache_hit:
        return

    kernel_name = ""
    meta_dict = {}
    if hasattr(metadata, "__dict__"):
        meta_dict = {k: str(v)[:100] for k, v in vars(metadata).items()}
        kernel_name = getattr(metadata, "name", "")
    elif isinstance(metadata, dict):
        meta_dict = {k: str(v)[:100] for k, v in metadata.items()}
        kernel_name = metadata.get("name", "")

    compile_time_ms = 0.0
    if times is not None:
        if hasattr(times, "total"):
            compile_time_ms = getattr(times, "total", 0.0) * 1000
        elif isinstance(times, (int, float)):
            compile_time_ms = float(times) * 1000

    _captured_triton.append({
        "kernel_name": kernel_name,
        "src_length": len(src),
        "metadata": meta_dict,
        "compile_time_ms": compile_time_ms,
    })
