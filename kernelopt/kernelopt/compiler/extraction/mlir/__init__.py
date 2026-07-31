"""MLIR-based analysis for cross-framework IR comparison.

Provides StableHLO and Linalg structural analysis via torch-mlir.
torch-mlir is an optional dependency.
"""

from kernelopt.compiler.extraction.mlir.stablehlo import (
    analyze_stablehlo,
    export_to_stablehlo,
)
from kernelopt.compiler.extraction.mlir.linalg_analysis import (
    analyze_linalg,
    export_to_linalg,
)

__all__ = [
    "analyze_stablehlo",
    "export_to_stablehlo",
    "analyze_linalg",
    "export_to_linalg",
]
