"""Norm → GEMM fusion pattern (stub).

Will detect triton_red/per_*norm* → extern_kernels.mm/addmm for
fused normalization + matrix multiply in a persistent-kernel style.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from kernelopt.fusion.graph import BufferGraph
from kernelopt.fusion.patterns import FusionPattern, PatternMatch


class NormGEMMFusion(FusionPattern):
    """Fuse normalization + GEMM (stub — not yet implemented)."""

    name = "norm_linear"
    priority = 8

    def match(
        self, graph: BufferGraph, kernel_sources: Dict[str, str]
    ) -> List[PatternMatch]:
        return []

    def build_prompt(self, match: PatternMatch) -> Tuple[str, str]:
        return ("", "")
