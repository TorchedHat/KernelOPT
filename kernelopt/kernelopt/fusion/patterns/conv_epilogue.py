"""Convolution epilogue detection pattern.

Detects extern_kernels.convolution followed by single-consumer
triton_poi/per/red kernels (typically BN + ReLU). This pattern does NOT
perform true fusion (cuDNN conv can't be fused with Triton BN) — instead
it annotates the pair so the per-kernel optimizer can skip the conv
and focus budget on the BN+ReLU epilogue.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from kernelopt.fusion.graph import BufferGraph, ExternKernelNode, KernelNode
from kernelopt.fusion.patterns import FusionPattern, PatternMatch
from kernelopt.fusion.patterns.base import (
    collect_external_io,
    get_node_by_name,
    is_model_output,
)


class ConvEpilogueFusion(FusionPattern):
    """Detect conv + BN/ReLU epilogue chains for annotation."""

    name = "conv_epilogue"
    priority = 10

    def match(
        self, graph: BufferGraph, kernel_sources: Dict[str, str]
    ) -> List[PatternMatch]:
        matches: List[PatternMatch] = []

        for enode in graph.extern_kernels:
            if enode.op != "convolution":
                continue
            if not enode.writes:
                continue

            write_buf = enode.writes[0]

            # Resolve alias
            alias_buf = None
            for alias, orig in graph.buf_aliases.items():
                if orig == write_buf:
                    alias_buf = alias
                    break

            # Find single consumer
            consumers = [
                c for (p, b, c) in graph.edges
                if b in (write_buf, alias_buf) and p == enode.name
            ]
            if len(consumers) != 1:
                continue

            consumer_name = consumers[0]
            consumer = get_node_by_name(consumer_name, graph)
            if consumer is None or not isinstance(consumer, KernelNode):
                continue

            # Must be a BN/norm/relu epilogue
            if consumer.kernel_type not in ("poi", "per", "red"):
                continue

            # Check name contains batch_norm indicators
            bn_indicators = (
                "batch_norm", "native_batch_norm", "_bn_", "norm",
                "relu", "softmax",
            )
            if not any(ind in consumer.name for ind in bn_indicators):
                continue

            if is_model_output(write_buf, graph):
                continue

            nodes = [enode, consumer]
            elim_bufs = [graph.buffers[write_buf]] if write_buf in graph.buffers else []
            elim_set = {write_buf}
            if alias_buf:
                elim_set.add(alias_buf)

            inputs, outputs = collect_external_io(nodes, elim_set, graph)
            sources = {consumer_name: kernel_sources.get(consumer_name, "")}

            matches.append(PatternMatch(
                pattern_name=self.name,
                pattern=self,
                nodes=nodes,
                eliminated_buffers=elim_bufs,
                input_buffers=inputs,
                output_buffers=outputs,
                kernel_sources=sources,
                metadata={
                    "conv_node": enode.name,
                    "epilogue_node": consumer_name,
                    "epilogue_type": consumer.kernel_type,
                    "annotation_only": True,  # Flag: don't generate fused kernel
                },
                estimated_profit=1.15,
            ))

        return matches

    def build_prompt(self, match: PatternMatch) -> Tuple[str, str]:
        # Conv epilogue is annotation-only — no LLM fusion generation
        return ("", "")
