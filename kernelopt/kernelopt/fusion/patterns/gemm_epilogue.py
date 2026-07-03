"""GEMM epilogue fusion pattern.

Detects extern_kernels.mm/addmm/bmm followed by single-consumer
triton_poi_* chains. Also matches triton_tem_* (Inductor GEMM templates)
followed by triton_poi_*.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from kernelopt.fusion.graph import BufferGraph, ExternKernelNode, KernelNode
from kernelopt.fusion.patterns import FusionPattern, PatternMatch
from kernelopt.fusion.patterns.base import (
    collect_external_io,
    get_buffer_size_bytes,
    get_node_by_name,
    is_model_output,
)


GEMM_EPILOGUE_SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer specializing in Triton GEMM fusion.

Your task: replace a GEMM operation (matrix multiply) and its pointwise \
epilogue with a SINGLE fused Triton kernel that computes the matmul and \
applies the epilogue (bias, activation, etc.) in one pass.

Rules:
1. The fused kernel must produce the same outputs as the original sequence.
2. Eliminate the intermediate GEMM output buffer — keep it in registers.
3. Preserve numerical correctness. Do NOT use allow_tf32=True or cast \
   operands to BF16. Keep full FP32 precision for tl.dot accumulation.
4. Return ONLY valid Python source code — no markdown, no explanation.
5. Use only ASCII characters — no Unicode math symbols.
6. Function name: fused_kernel. Wrapper: fused_kernel_call().
7. ALL computation in the @triton.jit kernel.

GEMM structure:
- Use tl.dot for matrix multiply with FP32 accumulator
- Tile the M and N dimensions with BLOCK_M and BLOCK_N
- Loop over K dimension with BLOCK_K
- After the K-loop, apply the epilogue (bias add, activation) to the \
  accumulator before storing

Example GEMM + bias + GELU fusion:
```
@triton.jit
def fused_gemm_gelu(A_ptr, B_ptr, bias_ptr, C_ptr,
                     M, N, K, stride_am, stride_ak,
                     stride_bk, stride_bn, stride_cm, stride_cn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                     mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        offs_k += BLOCK_K
    # Epilogue: bias + GELU
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    x = acc
    out = x * 0.5 * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             out.to(C_ptr.dtype.element_ty), mask=mask)
```

Include @triton.autotune with multiple BLOCK_M/N/K configs.
"""

_GEMM_OPS = {"mm", "addmm", "bmm", "bias_addmm"}


class GEMMEpilogueFusion(FusionPattern):
    """Fuse GEMM (extern_kernels or triton_tem) + pointwise epilogue."""

    name = "gemm_epilogue"
    priority = 10

    def match(
        self, graph: BufferGraph, kernel_sources: Dict[str, str]
    ) -> List[PatternMatch]:
        matches: List[PatternMatch] = []

        # --- Match extern_kernels GEMM → triton_poi chains ---
        for enode in graph.extern_kernels:
            if enode.op not in _GEMM_OPS:
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
            if consumer.kernel_type != "poi":
                continue

            # Don't fuse if the GEMM output is a model output
            if is_model_output(write_buf, graph):
                continue

            # Build match: extern GEMM + pointwise epilogue
            nodes = [enode, consumer]
            elim_buf_name = write_buf
            elim_bufs = [graph.buffers[elim_buf_name]] if elim_buf_name in graph.buffers else []
            elim_set = {elim_buf_name}
            if alias_buf:
                elim_set.add(alias_buf)

            inputs, outputs = collect_external_io(nodes, elim_set, graph)
            sources = {consumer_name: kernel_sources.get(consumer_name, "")}

            # Cost estimation
            profit = self._estimate_profit(enode, elim_bufs)

            matches.append(PatternMatch(
                pattern_name=self.name,
                pattern=self,
                nodes=nodes,
                eliminated_buffers=elim_bufs,
                input_buffers=inputs,
                output_buffers=outputs,
                kernel_sources=sources,
                metadata={"gemm_op": enode.op, "extern_node": enode.name},
                estimated_profit=profit,
            ))

        # --- Match triton_tem (GEMM template) → triton_poi chains ---
        for knode in graph.kernels:
            if knode.kernel_type != "tem":
                continue
            if not knode.writes:
                continue

            write_buf = knode.writes[0]
            alias_buf = None
            for alias, orig in graph.buf_aliases.items():
                if orig == write_buf:
                    alias_buf = alias
                    break

            consumers = [
                c for (p, b, c) in graph.edges
                if b in (write_buf, alias_buf) and p == knode.name
            ]
            if len(consumers) != 1:
                continue

            consumer_name = consumers[0]
            consumer = get_node_by_name(consumer_name, graph)
            if consumer is None or not isinstance(consumer, KernelNode):
                continue
            if consumer.kernel_type != "poi":
                continue

            if is_model_output(write_buf, graph):
                continue

            nodes = [knode, consumer]
            elim_bufs = [graph.buffers[write_buf]] if write_buf in graph.buffers else []
            elim_set = {write_buf}
            if alias_buf:
                elim_set.add(alias_buf)

            inputs, outputs = collect_external_io(nodes, elim_set, graph)
            sources = {
                knode.name: kernel_sources.get(knode.name, ""),
                consumer_name: kernel_sources.get(consumer_name, ""),
            }

            profit = 1.1  # template GEMM epilogue is usually profitable

            matches.append(PatternMatch(
                pattern_name=self.name,
                pattern=self,
                nodes=nodes,
                eliminated_buffers=elim_bufs,
                input_buffers=inputs,
                output_buffers=outputs,
                kernel_sources=sources,
                metadata={"gemm_op": "triton_tem", "template_node": knode.name},
                estimated_profit=profit,
            ))

        return matches

    def _estimate_profit(
        self, enode: ExternKernelNode, elim_bufs: list
    ) -> float:
        """Estimate fusion profitability from buffer sizes."""
        if not elim_bufs:
            return 1.05
        total_bytes = sum(get_buffer_size_bytes(b) for b in elim_bufs)
        # Large GEMM output → high savings
        if total_bytes > 4 * 1024 * 1024:  # > 4MB
            return 1.2
        elif total_bytes > 64 * 1024:  # > 64KB
            return 1.1
        else:
            return 0.8  # small GEMM, Triton unlikely to beat cuBLAS

    def build_prompt(self, match: PatternMatch) -> Tuple[str, str]:
        lines = []
        lines.append(f"## Fusion: {self.name}")
        lines.append(f"GEMM op: {match.metadata.get('gemm_op', 'unknown')}")
        lines.append("")

        lines.append("### Nodes to fuse (in execution order)")
        for i, node in enumerate(match.nodes):
            lines.append(f"Node {i}: {node.name} (type={getattr(node, 'kernel_type', getattr(node, 'op', 'extern'))})")
            lines.append(f"  reads: {node.reads}")
            lines.append(f"  writes: {node.writes}")
        lines.append("")

        lines.append("### Intermediate buffers to eliminate")
        for buf in match.eliminated_buffers:
            lines.append(f"  {buf.name}: shape={buf.shape}, strides={buf.strides}, dtype={buf.dtype}")
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
            "replaces the GEMM + epilogue. Return only valid Python source code."
        )

        return GEMM_EPILOGUE_SYSTEM_PROMPT, "\n".join(lines)
