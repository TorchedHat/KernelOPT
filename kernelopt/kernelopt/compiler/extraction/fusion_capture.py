"""Capture Inductor's fusion decisions via scheduler custom passes.

Uses _pre_fusion_custom_pass and _post_fusion_custom_pass hooks to
observe the scheduler node list before and after fusion. Also patches
WhyNoFuse to capture rejection reasons.

WARNING: These hooks are underscore-prefixed ("prototype stage").
If they break in a future PyTorch version, extraction degrades gracefully
(fusion data missing, but IR and codegen data still available).

References:
    - _pre_fusion_custom_pass: scheduler.py line 4259
    - _post_fusion_custom_pass: scheduler.py line 4287
    - WhyNoFuse: scheduler.py line 2094
    - FusedSchedulerNode: scheduler.py line 2690
"""

from __future__ import annotations

import threading
from typing import Any


_capture_store = threading.local()


def _get_store():
    if not hasattr(_capture_store, "pre_fusion_nodes"):
        _capture_store.pre_fusion_nodes = []
        _capture_store.post_fusion_nodes = []
        _capture_store.why_no_fuse_rejections = []
    return _capture_store


def install_fusion_hooks() -> bool:
    """Install fusion capture hooks into Inductor config.

    Call BEFORE torch.compile() in the extraction subprocess.
    Returns True if hooks were installed successfully.
    """
    try:
        import torch._inductor.config as _config
    except ImportError:
        return False

    store = _get_store()

    def _pre_fusion_pass(nodes):
        store.pre_fusion_nodes = []
        for node in nodes:
            try:
                store.pre_fusion_nodes.append({
                    "name": node.get_name(),
                    "names": list(node.get_operation_names()),
                    "is_reduction": node.is_reduction(),
                })
            except Exception:
                pass
        return nodes

    def _post_fusion_pass(nodes):
        try:
            from torch._inductor.scheduler import FusedSchedulerNode
        except ImportError:
            return nodes

        store.post_fusion_nodes = []
        for node in nodes:
            try:
                entry = {
                    "name": node.get_name(),
                    "names": list(node.get_operation_names()),
                    "is_fused": isinstance(node, FusedSchedulerNode),
                    "is_reduction": node.is_reduction(),
                }
                if isinstance(node, FusedSchedulerNode):
                    entry["snodes"] = [sn.get_name() for sn in node.snodes]
                store.post_fusion_nodes.append(entry)
            except Exception:
                pass
        return nodes

    if not hasattr(_config, "_pre_fusion_custom_pass"):
        return False

    _config._pre_fusion_custom_pass = _pre_fusion_pass
    _config._post_fusion_custom_pass = _post_fusion_pass

    _patch_why_no_fuse(store)
    return True


def _patch_why_no_fuse(store) -> None:
    """Wrap WhyNoFuse.__call__ to capture rejection reasons."""
    try:
        from torch._inductor.scheduler import WhyNoFuse
    except ImportError:
        return

    if not hasattr(WhyNoFuse, "__call__"):
        return

    _original_call = WhyNoFuse.__call__

    def _patched_call(self, reason: str, *args: Any) -> None:
        try:
            formatted = reason % args if args else reason
            store.why_no_fuse_rejections.append({
                "node1": self.name1,
                "node2": self.name2,
                "reason": formatted,
            })
        except Exception:
            pass
        return _original_call(self, reason, *args)

    WhyNoFuse.__call__ = _patched_call


_STRUCTURAL_KEYWORDS = frozenset({
    "device mismatch", "extern but not a triton",
    "different devices", "not a GPU node",
    "will create cycle", "data dependency",
})


def extract_fusion_data(graph_lowering=None) -> dict:
    """After compilation, process captured fusion data.

    Returns dict with: fusion_groups, fusion_rejections, intermediate_buffers.
    """
    store = _get_store()

    fusion_groups = []
    for node in store.post_fusion_nodes:
        if node.get("is_fused"):
            fusion_groups.append({
                "kernel_name": node["name"],
                "constituent_buffers": tuple(node.get("snodes", [])),
                "constituent_ops": (),
                "shared_bytes": 0,
            })

    fusion_rejections = []
    for rej in store.why_no_fuse_rejections[:50]:
        reason = rej.get("reason", "")
        is_structural = any(kw in reason.lower() for kw in _STRUCTURAL_KEYWORDS)
        is_heuristic = any(kw in reason.lower() for kw in ("score", "threshold", "too many"))

        fusion_rejections.append({
            "node1_name": rej["node1"],
            "node2_name": rej["node2"],
            "reason": reason,
            "near_threshold": is_heuristic and not is_structural,
            "fixable_by_manual_fusion": not is_structural,
        })

    intermediate_buffers = []
    if graph_lowering is not None:
        intermediate_buffers = _find_intermediate_buffers(
            graph_lowering, store.post_fusion_nodes
        )

    return {
        "fusion_groups": fusion_groups,
        "fusion_rejections": fusion_rejections,
        "intermediate_buffers": intermediate_buffers,
    }


def _find_intermediate_buffers(graph_lowering, post_fusion_nodes: list) -> list[dict]:
    """Identify buffers that flow between two separate kernels."""
    try:
        from torch._inductor import ir as _ir
    except ImportError:
        return []

    buf_producers: dict[str, str] = {}
    buf_consumers: dict[str, list[str]] = {}

    for node in post_fusion_nodes:
        node_name = node["name"]
        for op_name in node.get("names", []):
            buf = graph_lowering.name_to_buffer.get(op_name)
            if buf is None:
                continue
            buf_producers[op_name] = node_name
            try:
                rw = buf.get_read_writes() if hasattr(buf, "get_read_writes") else None
                if rw:
                    for dep in rw.reads:
                        if hasattr(dep, "name"):
                            buf_consumers.setdefault(dep.name, []).append(node_name)
            except Exception:
                pass

    results = []
    for buf_name, producer in buf_producers.items():
        consumers = buf_consumers.get(buf_name, [])
        external = [c for c in consumers if c != producer]
        if not external:
            continue

        buf = graph_lowering.name_to_buffer.get(buf_name)
        if buf is None:
            continue

        size_bytes = 0
        dtype_str = "unknown"
        try:
            layout = buf.layout
            numel = 1
            for s in layout.size:
                val = int(s)
                numel *= val
            size_bytes = numel * layout.dtype.itemsize
            dtype_str = str(layout.dtype)
        except Exception:
            pass

        results.append({
            "name": buf_name,
            "size_bytes": size_bytes,
            "dtype": dtype_str,
            "producer_kernel": producer,
            "consumer_kernels": tuple(set(external)),
            "lifetime_span": 0,
            "eliminable_by_fusion": len(set(external)) == 1,
        })

    return results
