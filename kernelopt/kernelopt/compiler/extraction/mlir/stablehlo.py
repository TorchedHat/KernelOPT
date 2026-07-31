"""StableHLO structural analysis via torch-mlir.

Uses torch_mlir.fx.export_and_import to convert PyTorch FX graphs
to StableHLO MLIR, then parses the text for structural metadata.

References:
    - torch-mlir FxImporter: python/torch_mlir/fx.py
    - StableHLO ops: https://github.com/openxla/stablehlo/blob/main/docs/spec.md
"""

from __future__ import annotations

import re
from typing import Any


def export_to_stablehlo(model: Any, example_inputs: tuple) -> str | None:
    """Export a PyTorch model to StableHLO text via torch-mlir.

    Args:
        model: A torch.nn.Module or exported program.
        example_inputs: Tuple of example input tensors.

    Returns:
        StableHLO module text, or None if torch-mlir unavailable.
    """
    try:
        import torch
        from torch_mlir.fx import export_and_import, OutputType
    except ImportError:
        return None

    try:
        exported = torch.export.export(model, example_inputs)
        module = export_and_import(exported, output_type=OutputType.STABLEHLO)
        return str(module)
    except Exception:
        return None


def analyze_stablehlo(hlo_text: str) -> dict:
    """Extract structural metadata from StableHLO module text.

    Returns:
        Dict with op_counts, dimension_numbers, memory_patterns, etc.
    """
    if not hlo_text:
        return {}

    op_counts = _count_ops(hlo_text)
    dim_numbers = _extract_dimension_numbers(hlo_text)
    patterns = _classify_patterns(op_counts)

    return {
        "op_counts": op_counts,
        "dimension_numbers": dim_numbers,
        "patterns": patterns,
        "total_ops": sum(op_counts.values()),
    }


def _count_ops(hlo_text: str) -> dict[str, int]:
    """Count StableHLO operations by type."""
    stablehlo_ops = [
        "stablehlo.dot_general",
        "stablehlo.convolution",
        "stablehlo.reduce",
        "stablehlo.broadcast_in_dim",
        "stablehlo.transpose",
        "stablehlo.reshape",
        "stablehlo.concatenate",
        "stablehlo.slice",
        "stablehlo.gather",
        "stablehlo.scatter",
        "stablehlo.select",
        "stablehlo.compare",
        "stablehlo.add",
        "stablehlo.multiply",
        "stablehlo.subtract",
        "stablehlo.divide",
        "stablehlo.maximum",
        "stablehlo.minimum",
        "stablehlo.exponential",
        "stablehlo.log",
        "stablehlo.tanh",
        "stablehlo.clamp",
        "stablehlo.convert",
        "stablehlo.constant",
        "stablehlo.iota",
        "stablehlo.dynamic_slice",
        "stablehlo.pad",
    ]

    counts = {}
    for op in stablehlo_ops:
        count = hlo_text.count(op)
        if count > 0:
            counts[op.replace("stablehlo.", "")] = count

    return counts


def _extract_dimension_numbers(hlo_text: str) -> list[dict]:
    """Extract dot_general dimension_numbers for matmul analysis."""
    results = []
    for match in re.finditer(
        r'dot_dimension_numbers\s*=\s*#stablehlo\.dot<'
        r'(?:lhs_batching_dimensions\s*=\s*\[([^\]]*)\]\s*,\s*)?'
        r'rhs_batching_dimensions\s*=\s*\[([^\]]*)\]\s*,\s*'
        r'lhs_contracting_dimensions\s*=\s*\[([^\]]*)\]\s*,\s*'
        r'rhs_contracting_dimensions\s*=\s*\[([^\]]*)\]',
        hlo_text,
    ):
        results.append({
            "lhs_batch": [int(x) for x in match.group(1).split(",") if x.strip()] if match.group(1) else [],
            "rhs_batch": [int(x) for x in match.group(2).split(",") if x.strip()] if match.group(2) else [],
            "lhs_contract": [int(x) for x in match.group(3).split(",") if x.strip()] if match.group(3) else [],
            "rhs_contract": [int(x) for x in match.group(4).split(",") if x.strip()] if match.group(4) else [],
        })

    return results


def _classify_patterns(op_counts: dict[str, int]) -> list[str]:
    """Classify the computation pattern from op counts."""
    patterns = []

    if op_counts.get("dot_general", 0) > 0:
        patterns.append("compute_bound_matmul")
    if op_counts.get("convolution", 0) > 0:
        patterns.append("compute_bound_conv")
    if op_counts.get("reduce", 0) > 0:
        patterns.append("reduction")

    elementwise = sum(
        op_counts.get(op, 0)
        for op in ("add", "multiply", "subtract", "divide", "maximum", "minimum")
    )
    memory_ops = sum(
        op_counts.get(op, 0)
        for op in ("broadcast_in_dim", "transpose", "reshape", "concatenate", "slice", "gather")
    )

    if elementwise > 0 and op_counts.get("dot_general", 0) == 0:
        patterns.append("memory_bound_elementwise")
    if memory_ops > elementwise and elementwise > 0:
        patterns.append("memory_layout_heavy")

    return patterns
