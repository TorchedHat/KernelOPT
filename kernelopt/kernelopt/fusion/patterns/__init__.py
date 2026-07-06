"""Pattern-based cross-kernel fusion framework.

Provides a registry of FusionPattern subclasses that detect higher-level
computational patterns in the Inductor kernel graph (GEMM+epilogue,
residual+norm, pointwise chains, etc.) and generate pattern-specific
fused kernels via targeted LLM prompts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from kernelopt.fusion.graph import BufferGraph, BufferNode, ExternKernelNode, KernelNode


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class PatternMatch:
    """A single detected fusion opportunity."""

    pattern_name: str
    pattern: "FusionPattern"  # the pattern that created this match
    nodes: List[Union[KernelNode, ExternKernelNode]]  # in execution order
    eliminated_buffers: List[BufferNode]
    input_buffers: List[str]
    output_buffers: List[str]
    kernel_sources: Dict[str, str]   # name -> source for Triton nodes
    metadata: Dict[str, Any] = field(default_factory=dict)
    estimated_profit: float = 1.0


class FusionPattern(ABC):
    """Base class for model-level fusion patterns."""

    name: str = ""
    priority: int = 0  # higher = matched first

    @abstractmethod
    def match(
        self, graph: BufferGraph, kernel_sources: Dict[str, str]
    ) -> List[PatternMatch]:
        """Find all instances of this pattern in the graph."""

    @abstractmethod
    def build_prompt(self, match: PatternMatch) -> Tuple[str, str]:
        """Return (system_prompt, human_prompt) for LLM fusion generation."""


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------


class PatternRegistry:
    """Registry of fusion patterns with overlap resolution."""

    def __init__(self) -> None:
        self._patterns: List[FusionPattern] = []

    def register(self, pattern: FusionPattern) -> None:
        self._patterns.append(pattern)
        self._patterns.sort(key=lambda p: p.priority, reverse=True)

    def match_all(
        self,
        graph: BufferGraph,
        kernel_sources: Dict[str, str],
    ) -> List[PatternMatch]:
        """Run all registered patterns and resolve overlaps.

        Returns non-overlapping matches sorted by execution order.
        """
        all_matches: List[PatternMatch] = []

        for pattern in self._patterns:
            matches = pattern.match(graph, kernel_sources)
            all_matches.extend(matches)

        if not all_matches:
            return []

        # Sort by estimated profit descending
        all_matches.sort(key=lambda m: m.estimated_profit, reverse=True)

        # Greedy overlap resolution: skip matches sharing nodes
        selected: List[PatternMatch] = []
        claimed_nodes: set = set()

        for match in all_matches:
            node_names = {n.name for n in match.nodes}
            if node_names & claimed_nodes:
                continue  # overlaps with already-selected match
            if match.estimated_profit < 1.0:
                continue  # not profitable
            selected.append(match)
            claimed_nodes |= node_names

        # Sort selected by execution order
        if graph.node_order:
            order_map = {name: i for i, name in enumerate(graph.node_order)}
            selected.sort(
                key=lambda m: min(
                    order_map.get(n.name, float("inf")) for n in m.nodes
                )
            )

        return selected


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


def default_registry() -> PatternRegistry:
    """Create a registry with all built-in patterns."""
    from kernelopt.fusion.patterns.conv_epilogue import ConvEpilogueFusion
    from kernelopt.fusion.patterns.gemm_epilogue import GEMMEpilogueFusion
    from kernelopt.fusion.patterns.pointwise_chain import PointwiseChainFusion
    from kernelopt.fusion.patterns.residual_norm import ResidualNormFusion

    # Stubs — import only if they exist and have real implementations
    try:
        from kernelopt.fusion.patterns.attention import AttentionFusion
    except ImportError:
        AttentionFusion = None
    try:
        from kernelopt.fusion.patterns.norm_linear import NormGEMMFusion
    except ImportError:
        NormGEMMFusion = None
    try:
        from kernelopt.fusion.patterns.multi_gemm import MultiGEMMFusion
    except ImportError:
        MultiGEMMFusion = None

    reg = PatternRegistry()
    reg.register(GEMMEpilogueFusion())
    reg.register(ConvEpilogueFusion())
    reg.register(ResidualNormFusion())
    reg.register(PointwiseChainFusion())

    if AttentionFusion is not None:
        reg.register(AttentionFusion())
    if NormGEMMFusion is not None:
        reg.register(NormGEMMFusion())
    if MultiGEMMFusion is not None:
        reg.register(MultiGEMMFusion())

    return reg
