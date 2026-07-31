"""Track AOTAutograd decomposition decisions.

There is no per-op callback when a decomposition fires. Instead we:
1. Inspect the decomposition table to know which ops WILL be decomposed
2. Diff the pre-decomposition graph (from dynamo_capture) against the
   post-decomposition graph (from post_grad_custom_pre_pass) to see
   which ops were actually decomposed and into what.

References:
    - select_decomp_table(): torch/_inductor/decomposition.py line 1168
    - make_fx with decomps: torch/_functorch/_aot_autograd/graph_capture.py line 131
    - compile_fx decompositions parameter: torch/_inductor/compile_fx.py line 2889
"""

from __future__ import annotations


def get_decomposition_table_info() -> dict:
    """Inspect the default Inductor decomposition table.

    Returns summary of which ops have decompositions and how many total.
    """
    try:
        from torch._inductor.decomposition import select_decomp_table
        table = select_decomp_table()
    except ImportError:
        return {"total_entries": 0, "compound_ops": []}

    compound_ops = []
    target_ops = {
        "layer_norm", "native_layer_norm", "group_norm", "native_group_norm",
        "batch_norm", "native_batch_norm", "_native_batch_norm_legit",
        "gelu", "silu", "softmax", "_softmax", "log_softmax",
        "linear", "embedding", "dropout",
    }

    for key in table:
        name = str(key).lower()
        if any(op in name for op in target_ops):
            compound_ops.append(str(key))

    return {
        "total_entries": len(table),
        "compound_ops": compound_ops,
    }


def diff_pre_post_decomp(
    pre_decomp_ops: list[dict],
    post_decomp_node_types: list[str],
) -> list[dict]:
    """Diff pre-decomposition ops against post-decomposition node types.

    Args:
        pre_decomp_ops: From dynamo_capture (list of {op_name, is_compound, ...})
        post_decomp_node_types: Node types from ir_capture (Pointwise, Reduction, etc.)

    Returns:
        List of AOTDecompInfo-compatible dicts for compound ops that were decomposed.
    """
    decomp_records = []

    for op in pre_decomp_ops:
        if not op.get("is_compound"):
            continue

        op_name = op["op_name"]
        decomposed_into = _known_decompositions.get(
            _normalize_op_name(op_name), None
        )

        if decomposed_into is not None:
            decomp_records.append({
                "original_op": op_name,
                "decomposed_into": tuple(decomposed_into),
                "num_intermediates": max(0, len(decomposed_into) - 1),
                "decomposition_source": "inductor_decompositions",
            })

    return decomp_records


def _normalize_op_name(name: str) -> str:
    """Normalize op name for lookup."""
    name = name.lower()
    for prefix in ("aten.", "torch.ops.aten.", "torch._c._nn."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in (".default", ".out"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


_known_decompositions: dict[str, list[str]] = {
    "native_layer_norm": ["var_mean", "rsqrt", "sub", "mul", "add"],
    "layer_norm": ["var_mean", "rsqrt", "sub", "mul", "add"],
    "gelu": ["mul", "erf", "add", "mul"],
    "native_group_norm": ["reshape", "var_mean", "rsqrt", "sub", "mul", "add", "reshape"],
    "group_norm": ["reshape", "var_mean", "rsqrt", "sub", "mul", "add", "reshape"],
    "native_batch_norm": ["mean", "var", "sub", "rsqrt", "mul", "add"],
    "_native_batch_norm_legit": ["mean", "var", "sub", "rsqrt", "mul", "add"],
    "softmax": ["max", "sub", "exp", "sum", "div"],
    "_softmax": ["max", "sub", "exp", "sum", "div"],
    "log_softmax": ["max", "sub", "exp", "sum", "log", "sub"],
    "linear": ["addmm"],
    "silu": [],  # NOT decomposed (excluded from table)
}
