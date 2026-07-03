"""Residual + Normalization fusion pattern.

Detects triton_poi_*add* followed by triton_red/per_*norm* — common in
ResNet bottleneck blocks and transformer residual connections.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from kernelopt.fusion.graph import BufferGraph, KernelNode
from kernelopt.fusion.patterns import FusionPattern, PatternMatch
from kernelopt.fusion.patterns.base import (
    collect_external_io,
    get_node_by_name,
    is_model_output,
)


RESIDUAL_NORM_SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer specializing in Triton kernel fusion.

Your task: fuse a residual addition kernel and a subsequent normalization \
kernel into a SINGLE kernel that computes both operations in one pass.

Rules:
1. Same external inputs/outputs as the original pair.
2. Eliminate the intermediate buffer between add and norm.
3. Preserve numerical correctness — accumulate in FP32.
4. Return ONLY valid Python source code — no markdown, no explanation.
5. Use only ASCII characters — no Unicode math symbols.
6. Function name: fused_kernel. Wrapper: fused_kernel_call().
7. ALL computation in the @triton.jit kernel.

Fusion technique for residual + LayerNorm:
- Load residual input and main input
- Compute x = residual + main (in registers)
- First pass: compute mean = sum(x) / N
- Second pass: compute var = sum((x - mean)^2) / N
- Normalize: out = (x - mean) / sqrt(var + eps) * weight + bias
- Store output

For numerical stability, accumulate mean and variance in FP32:
    mean_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    var_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
"""


class ResidualNormFusion(FusionPattern):
    """Fuse residual add + normalization."""

    name = "residual_norm"
    priority = 7

    def match(
        self, graph: BufferGraph, kernel_sources: Dict[str, str]
    ) -> List[PatternMatch]:
        matches: List[PatternMatch] = []

        # Find reduction/persistent kernels that contain "norm"
        norm_kernels = [
            k for k in graph.kernels
            if k.kernel_type in ("red", "per")
            and any(
                ind in k.name
                for ind in ("norm", "layer_norm", "batch_norm", "native_batch_norm")
            )
        ]

        for norm_k in norm_kernels:
            # Check each buffer this norm kernel reads
            for buf in norm_k.reads:
                resolved = graph.buf_aliases.get(buf, buf)

                # Find the producer of this buffer
                producers = [
                    p for (p, b, c) in graph.edges
                    if b in (buf, resolved) and c == norm_k.name
                ]
                if len(producers) != 1:
                    continue

                prod_name = producers[0]
                prod_node = get_node_by_name(prod_name, graph)
                if prod_node is None or not isinstance(prod_node, KernelNode):
                    continue

                # Producer must be a pointwise kernel with "add" in the name
                if prod_node.kernel_type != "poi":
                    continue
                if "add" not in prod_node.name:
                    continue

                # The intermediate buffer must have a single consumer
                all_consumers = [
                    c for (p, b, c) in graph.edges
                    if b in (buf, resolved) and p == prod_name
                ]
                if len(all_consumers) != 1:
                    continue

                if is_model_output(resolved, graph):
                    continue

                nodes = [prod_node, norm_k]
                elim_bufs = [graph.buffers[resolved]] if resolved in graph.buffers else []
                elim_set = {resolved, buf}

                inputs, outputs = collect_external_io(nodes, elim_set, graph)
                sources = {
                    prod_name: kernel_sources.get(prod_name, ""),
                    norm_k.name: kernel_sources.get(norm_k.name, ""),
                }

                matches.append(PatternMatch(
                    pattern_name=self.name,
                    pattern=self,
                    nodes=nodes,
                    eliminated_buffers=elim_bufs,
                    input_buffers=inputs,
                    output_buffers=outputs,
                    kernel_sources=sources,
                    metadata={
                        "add_node": prod_name,
                        "norm_node": norm_k.name,
                        "norm_type": norm_k.kernel_type,
                    },
                    estimated_profit=1.15,
                ))
                break  # Only one match per norm kernel

        return matches

    def build_prompt(self, match: PatternMatch) -> Tuple[str, str]:
        lines = []
        lines.append(f"## Fusion: {self.name}")
        lines.append(f"Residual add ({match.metadata.get('add_node', '?')}) + "
                     f"Normalization ({match.metadata.get('norm_node', '?')})")
        lines.append("")

        lines.append("### Kernels to fuse (in execution order)")
        for i, node in enumerate(match.nodes):
            lines.append(f"Kernel {i}: {node.name} (type={node.kernel_type})")
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
            "computes the residual add + normalization in one pass. "
            "Return only valid Python source code."
        )

        return RESIDUAL_NORM_SYSTEM_PROMPT, "\n".join(lines)
