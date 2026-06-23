"""Multi-head attention block fusion pattern (stub).

Will detect QKV projection → Q@K → softmax → @V → output projection
chains for FlashAttention-style fusion.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from kernelopt.fusion.graph import BufferGraph
from kernelopt.fusion.patterns import FusionPattern, PatternMatch


class AttentionFusion(FusionPattern):
    """Fuse multi-head attention blocks (stub — not yet implemented)."""

    name = "attention"
    priority = 12

    def match(
        self, graph: BufferGraph, kernel_sources: Dict[str, str]
    ) -> List[PatternMatch]:
        return []

    def build_prompt(self, match: PatternMatch) -> Tuple[str, str]:
        return ("", "")
