"""Capture pre-decomposition FX graph from TorchDynamo.

Uses config.pre_grad_custom_pass (config.py line 313) which runs
BEFORE AOTAutograd applies decompositions. At this point the graph
still has high-level ops like aten.layer_norm, aten.gelu, aten.linear.

Also extracts graph break information via torch._dynamo.explain()
or the global graph_break_reasons list.

References:
    - pre_grad_custom_pass: torch/_inductor/config.py line 313
    - GraphCompileReason: torch/_dynamo/output_graph.py line 333
    - graph_break_reasons: torch/_dynamo/utils.py line 3855
    - torch._dynamo.explain(): torch/_dynamo/eval_frame.py line 1869
"""

from __future__ import annotations

import threading
from typing import Any


_dynamo_store = threading.local()


def _get_store():
    if not hasattr(_dynamo_store, "pre_decomp_ops"):
        _dynamo_store.pre_decomp_ops = []
    return _dynamo_store


def install_dynamo_capture_hook() -> bool:
    """Install a pre-grad pass that captures the pre-decomposition graph.

    Call BEFORE torch.compile() in the extraction subprocess.
    Returns True if the hook was installed.
    """
    try:
        import torch._inductor.config as _config
    except ImportError:
        return False

    if not hasattr(_config, "pre_grad_custom_pass"):
        return False

    store = _get_store()

    def _capture_pre_decomp(graph):
        """Non-mutating pass: capture op names, return graph unchanged."""
        store.pre_decomp_ops = []
        try:
            for node in graph.nodes:
                if node.op == "call_function":
                    target = node.target
                    op_name = getattr(target, "__name__", str(target))
                    if hasattr(target, "overloadpacket"):
                        op_name = str(target.overloadpacket)

                    in_shapes = []
                    for arg in node.args:
                        if hasattr(arg, "meta") and "val" in arg.meta:
                            val = arg.meta["val"]
                            if hasattr(val, "shape"):
                                in_shapes.append(tuple(int(s) for s in val.shape))

                    out_shape = ()
                    if "val" in node.meta:
                        val = node.meta["val"]
                        if hasattr(val, "shape"):
                            out_shape = tuple(int(s) for s in val.shape)

                    store.pre_decomp_ops.append({
                        "op_name": op_name,
                        "input_shapes": tuple(in_shapes),
                        "output_shape": out_shape,
                        "is_compound": _is_compound_op(op_name),
                    })
        except Exception:
            pass
        return graph

    _config.pre_grad_custom_pass = _capture_pre_decomp
    return True


_COMPOUND_OPS = frozenset({
    "layer_norm", "native_layer_norm", "group_norm", "native_group_norm",
    "batch_norm", "native_batch_norm", "_native_batch_norm_legit",
    "softmax", "_softmax", "log_softmax", "_log_softmax",
    "gelu", "silu", "mish",
    "linear", "conv2d", "conv1d", "conv3d",
    "conv_transpose2d", "conv_transpose1d", "conv_transpose3d",
    "scaled_dot_product_attention",
})


def _is_compound_op(op_name: str) -> bool:
    """Check if an op is compound (will be decomposed by AOTAutograd)."""
    name_lower = op_name.lower()
    return any(c in name_lower for c in _COMPOUND_OPS)


def extract_dynamo_graph_info() -> dict:
    """Extract captured pre-decomposition graph info.

    Call AFTER torch.compile() has run.
    Returns dict suitable for constructing DynamoGraphInfo.
    """
    store = _get_store()
    return {
        "original_ops": tuple(store.pre_decomp_ops),
        "graph_breaks": (),
        "num_subgraphs": 1,
        "unsupported_ops": (),
    }


def extract_graph_breaks(model_fn, example_inputs) -> dict:
    """Use torch._dynamo.explain() to get graph break info.

    This runs a SEPARATE Dynamo trace (not the main compilation).
    Only call if graph break analysis is needed.
    """
    try:
        import torch._dynamo
        explanation = torch._dynamo.explain(model_fn)(*example_inputs)

        breaks = []
        for reason_obj in getattr(explanation, "break_reasons", []):
            location = ""
            if hasattr(reason_obj, "user_stack") and reason_obj.user_stack:
                frame = reason_obj.user_stack[-1]
                location = f"{frame.filename}:{frame.lineno}"
            breaks.append({
                "reason": reason_obj.reason,
                "location": location,
                "ops_before": 0,
                "ops_after": 0,
            })

        return {
            "graph_breaks": tuple(breaks),
            "num_subgraphs": getattr(explanation, "graph_count", 1),
            "graph_break_count": getattr(explanation, "graph_break_count", 0),
        }
    except Exception:
        return {"graph_breaks": (), "num_subgraphs": 1, "graph_break_count": 0}
