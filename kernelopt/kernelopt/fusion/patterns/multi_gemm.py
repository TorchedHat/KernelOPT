"""Multi-GEMM same-input fusion pattern (stub).

Will detect multiple extern_kernels.mm/addmm calls that read the same
input buffer (e.g., QKV projections) and replace with a single
concatenated GEMM.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from kernelopt.fusion.graph import BufferGraph
from kernelopt.fusion.patterns import FusionPattern, PatternMatch


class MultiGEMMFusion(FusionPattern):
    """Fuse multiple GEMMs reading same input (stub — not yet implemented)."""

    name = "multi_gemm"
    priority = 9

    def match(
        self, graph: BufferGraph, kernel_sources: Dict[str, str]
    ) -> List[PatternMatch]:
        return []

    def build_prompt(self, match: PatternMatch) -> Tuple[str, str]:
        return ("", "")
