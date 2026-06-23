"""Pointwise chain fusion pattern.

Detects chains of triton_poi_* kernels connected by single-consumer
intermediate buffers. This migrates the existing logic from detect.py
into the pattern framework.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from kernelopt.fusion.graph import BufferGraph, BufferNode, KernelNode
from kernelopt.fusion.patterns import FusionPattern, PatternMatch
from kernelopt.fusion.patterns.base import (
    collect_external_io,
    get_node_by_name,
    is_model_output,
)


POINTWISE_CHAIN_SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer specializing in Triton kernel fusion.

Your task: given N adjacent pointwise Triton kernels connected by intermediate \
buffers, produce a SINGLE fused Triton kernel that eliminates those buffers.

Rules:
1. The fused kernel must accept the same external inputs and produce the same \
   external outputs as the original kernel sequence.
2. Eliminate ALL intermediate buffers — keep their values in registers.
3. Preserve numerical correctness: same dtypes, same computation, no precision loss.
4. Return ONLY valid Python source code — no markdown fences, no explanation.
5. Use only ASCII characters — no Unicode math symbols.
6. The function name must be exactly: fused_kernel
7. Include fused_kernel_call() wrapper for grid setup and launch.
8. ALL computation in the @triton.jit kernel. No torch.nn.* in the wrapper.

Fusion technique: merge the computation bodies of all kernels. Load each \
external input once, apply all operations in sequence keeping intermediates \
in registers, store each external output once.
"""


class PointwiseChainFusion(FusionPattern):
    """Fuse chains of adjacent pointwise Triton kernels."""

    name = "pointwise_chain"
    priority = 5

    def match(
        self, graph: BufferGraph, kernel_sources: Dict[str, str]
    ) -> List[PatternMatch]:
        # Build producer/consumer maps for fusible buffers
        producer_of: Dict[str, str] = {}
        consumer_of: Dict[str, str] = {}

        for prod, buf, cons in graph.edges:
            if buf in producer_of:
                continue  # multiple producers — skip
            producer_of[buf] = prod

            if buf in consumer_of:
                consumer_of[buf] = "__multiple__"
            else:
                consumer_of[buf] = cons

        # Find fusible buffers: single producer, single consumer, not model output
        fusible = set()
        for buf in producer_of:
            if consumer_of.get(buf, "__multiple__") == "__multiple__":
                continue
            if is_model_output(buf, graph):
                continue
            fusible.add(buf)

        # Build adjacency: producer -> [(buffer, consumer)]
        next_kernel: Dict[str, List[Tuple[str, str]]] = {}
        prev_kernel: Dict[str, str] = {}

        for buf in fusible:
            prod = producer_of[buf]
            cons = consumer_of[buf]
            next_kernel.setdefault(prod, []).append((buf, cons))
            prev_kernel[cons] = prod

        # Only consider poi→poi chains
        poi_kernels = {k.name for k in graph.kernels if k.kernel_type == "poi"}

        # Find chain starts: poi kernels that don't consume a fusible buffer from another poi
        chain_starts = []
        for k in graph.kernels:
            if k.kernel_type != "poi":
                continue
            if k.name in prev_kernel and prev_kernel[k.name] in poi_kernels:
                continue
            if k.name in next_kernel:
                chain_starts.append(k.name)

        # DFS enumerate chains
        matches: List[PatternMatch] = []

        def _dfs(
            current: str,
            chain_nodes: List[str],
            chain_bufs: List[str],
            visited: set,
        ) -> None:
            if current not in next_kernel:
                if len(chain_nodes) >= 2:
                    _make_match(chain_nodes[:], chain_bufs[:])
                return

            extended = False
            for buf, nxt in next_kernel[current]:
                if nxt not in poi_kernels:
                    continue
                if nxt in visited:
                    continue  # cycle detection
                chain_nodes.append(nxt)
                chain_bufs.append(buf)
                visited.add(nxt)
                _dfs(nxt, chain_nodes, chain_bufs, visited)
                visited.discard(nxt)
                chain_nodes.pop()
                chain_bufs.pop()
                extended = True

            if not extended and len(chain_nodes) >= 2:
                _make_match(chain_nodes[:], chain_bufs[:])

        def _make_match(node_names: List[str], buf_names: List[str]) -> None:
            nodes = []
            for nn in node_names:
                node = get_node_by_name(nn, graph)
                if node is not None:
                    nodes.append(node)

            elim_bufs = [graph.buffers[b] for b in buf_names if b in graph.buffers]
            elim_set = set(buf_names)
            inputs, outputs = collect_external_io(nodes, elim_set, graph)

            sources = {nn: kernel_sources.get(nn, "") for nn in node_names}

            profit = 1.0 + 0.05 * len(elim_bufs)

            matches.append(PatternMatch(
                pattern_name=self.name,
                pattern=self,
                nodes=nodes,
                eliminated_buffers=elim_bufs,
                input_buffers=inputs,
                output_buffers=outputs,
                kernel_sources=sources,
                metadata={"chain_length": len(nodes)},
                estimated_profit=profit,
            ))

        for start in chain_starts:
            _dfs(start, [start], [], {start})

        return matches

    def build_prompt(self, match: PatternMatch) -> Tuple[str, str]:
        lines = []
        lines.append(f"## Fusion: {self.name}")
        lines.append(f"Chain of {len(match.nodes)} pointwise kernels.")
        lines.append("")

        lines.append("### Kernels to fuse (in execution order)")
        for i, node in enumerate(match.nodes):
            lines.append(f"Kernel {i}: {node.name}")
            lines.append(f"  reads: {node.reads}")
            lines.append(f"  writes: {node.writes}")
        lines.append("")

        lines.append("### Intermediate buffers to eliminate")
        for buf in match.eliminated_buffers:
            lines.append(f"  {buf.name}: shape={buf.shape}, dtype={buf.dtype}")
        lines.append("")

        lines.append("### External inputs")
        for inp in match.input_buffers:
            lines.append(f"  {inp}")
        lines.append("")

        lines.append("### External outputs")
        for out in match.output_buffers:
            lines.append(f"  {out}")
        lines.append("")

        lines.append("### Kernel source code")
        for name, src in match.kernel_sources.items():
            lines.append(f"#### {name}")
            lines.append(src if src else "(source not available)")
            lines.append("")

        lines.append(
            "Produce a single fused Triton kernel named `fused_kernel` that "
            "replaces all of the above. Return only valid Python source code."
        )

        return POINTWISE_CHAIN_SYSTEM_PROMPT, "\n".join(lines)
