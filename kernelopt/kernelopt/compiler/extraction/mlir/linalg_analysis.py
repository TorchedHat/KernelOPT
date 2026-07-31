"""Linalg dialect analysis for memory access pattern classification.

When torch-mlir exports to Linalg, the iterator_types and affine_maps
reveal the computation structure at a level closer to the hardware.

References:
    - Linalg dialect: https://mlir.llvm.org/docs/Dialects/Linalg/
    - torch-mlir Linalg output: OutputType.LINALG_ON_TENSORS
"""

from __future__ import annotations

import re
from typing import Any


def export_to_linalg(model: Any, example_inputs: tuple) -> str | None:
    """Export a PyTorch model to Linalg dialect via torch-mlir.

    Args:
        model: A torch.nn.Module or exported program.
        example_inputs: Tuple of example input tensors.

    Returns:
        Linalg module text, or None if torch-mlir unavailable.
    """
    try:
        import torch
        from torch_mlir.fx import export_and_import, OutputType
    except ImportError:
        return None

    try:
        exported = torch.export.export(model, example_inputs)
        module = export_and_import(exported, output_type=OutputType.LINALG_ON_TENSORS)
        return str(module)
    except Exception:
        return None


def analyze_linalg(linalg_text: str) -> dict:
    """Extract structural metadata from Linalg module text.

    Returns:
        Dict with iterator_types, access_patterns, fusion_candidates.
    """
    if not linalg_text:
        return {}

    iterator_info = _extract_iterator_types(linalg_text)
    access_patterns = _extract_access_patterns(linalg_text)

    return {
        "generic_ops": len(iterator_info),
        "iterator_types": iterator_info,
        "access_patterns": access_patterns,
    }


def _extract_iterator_types(linalg_text: str) -> list[dict]:
    """Extract iterator_types from linalg.generic operations."""
    results = []

    for match in re.finditer(
        r'iterator_types\s*=\s*\[([^\]]+)\]',
        linalg_text,
    ):
        types_str = match.group(1)
        types = [t.strip().strip('"') for t in types_str.split(",")]
        parallel_count = types.count("parallel")
        reduction_count = types.count("reduction")

        classification = "elementwise"
        if reduction_count > 0 and parallel_count > 0:
            classification = "reduction"
        elif reduction_count > 0:
            classification = "full_reduction"

        results.append({
            "types": types,
            "parallel_dims": parallel_count,
            "reduction_dims": reduction_count,
            "classification": classification,
        })

    return results


def _extract_access_patterns(linalg_text: str) -> list[str]:
    """Identify memory access patterns from affine_maps."""
    patterns = []

    if "affine_map<(d0, d1) -> (d0, d1)>" in linalg_text:
        patterns.append("contiguous_2d")
    if "affine_map<(d0, d1) -> (d1, d0)>" in linalg_text:
        patterns.append("transpose_2d")
    if re.search(r"affine_map<\([^)]+\) -> \(0\)>", linalg_text):
        patterns.append("scalar_broadcast")
    if re.search(r"affine_map<\([^)]+\) -> \(d\d+\)>", linalg_text):
        patterns.append("vector_broadcast")

    return patterns
