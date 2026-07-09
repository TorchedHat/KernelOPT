"""LLM-based Triton kernel generation from traced computation graphs.

Given a kernel group (ATen ops + shapes + kernel type), the LLM writes a
clean standalone Triton kernel with:
  - @triton.autotune + @triton.jit decorated kernel function
  - kernel_function(*inputs) Python wrapper
  - get_inputs() returning properly shaped tensors
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from kernelopt.constants import MAX_SOURCE_CHARS
from kernelopt.llm import build_llm
from kernelopt.pipeline_ui import pipe_print


CODEGEN_SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer. You write high-performance Triton kernels
from PyTorch operation specifications.

Given a description of what computation to implement (ATen ops, tensor shapes,
dtypes), you produce a complete, runnable Python file containing:

1. A @triton.jit kernel (optionally with @triton.autotune)
2. A kernel_function(*inputs) wrapper that:
   - Takes PyTorch tensors as positional arguments
   - Allocates output tensors
   - Launches the kernel with correct grid
   - Returns the output tensor(s)
3. A get_inputs() function returning sample input tensors

RULES:
- Import only: torch, triton, triton.language as tl
- Use triton.language.extra.libdevice for math functions (not tl.sqrt, tl.tanh)
- Use float32 accumulators for numerical stability in reductions and matmuls
- Include boundary masks for ALL tl.load and tl.store calls
- Include @triton.autotune with at least 4 configs varying tile sizes and num_warps
- CRITICAL — AUTOTUNE + MASK: When using @triton.autotune with variable BLOCK sizes,
  EVERY tl.load and tl.store MUST have a mask= argument. Autotune configs may set BLOCK
  larger than the tensor dimension — threads beyond valid range will compute invalid
  indices and cause CUDA illegal memory access. Never mix masked and unmasked accesses.
- The kernel must produce results matching PyTorch eager mode (use torch.allclose with rtol=1e-3, atol=1e-5)
- Return ONLY valid Python code. No markdown fences. No explanations outside comments.
- The file must be self-contained and runnable: python file.py --profile-only

CRITICAL TRITON PATTERNS:
- For REDUCTION kernels (softmax, layernorm, sum, max): use 1D tiling with one
  row per program (program_id(0) = row index). Iterate over the reduction
  dimension in a loop with a 1D BLOCK_N. Do NOT use 2D tiling with BLOCK_M=1
  for reductions — this triggers a Triton compiler bug.
- For POINTWISE kernels: use 1D flat indexing with BLOCK_SIZE elements per program.
- For GEMM/MATMUL kernels: use 2D tiling with BLOCK_M x BLOCK_N output tile and
  BLOCK_K reduction dimension. Use tl.dot for the inner product.

CRITICAL — DO NOT REPLACE CUBLAS/CUDNN WITH TRITON:
When the ATen ops include matmul (mm, addmm, bmm), linear, or convolution
(conv2d, conv_transpose2d, conv3d), Inductor calls cuBLAS/cuDNN for these via
extern_kernels. These library calls are FASTER than any hand-written Triton
matmul or convolution. Do NOT rewrite them as tl.dot loops.

Instead, structure your kernel as:
1. KEEP the matmul/conv as a standard PyTorch call (torch.mm, F.conv2d, etc.)
2. FUSE ONLY the epilogue operations (activation, normalization, scaling,
   pooling) into a Triton kernel that reads the matmul/conv output and
   applies the epilogue in one pass.

Example — Linear + GELU + Dropout:
  BAD:  Write a full Triton matmul with tl.dot (slower than cuBLAS)
  GOOD: Call torch.mm(x, weight.T) + bias, then fuse GELU + dropout in Triton

Example — Conv2d + BatchNorm + ReLU:
  BAD:  Write a full Triton convolution (much slower than cuDNN)
  GOOD: Call F.conv2d(x, weight, bias), then fuse batch_norm + relu in Triton

The ONLY case where Triton matmul beats cuBLAS is when the algorithm is
fundamentally different (e.g., diagonal matmul, sparse matmul, flash attention).
For standard dense matmul/conv, always use the library call.

BEATING TORCH.COMPILE (INDUCTOR):
torch.compile (Inductor) generates good but not optimal code. Your goal is to
beat it. Here are patterns where hand-written Triton consistently wins:

1. EPILOGUE FUSION: Inductor calls cuBLAS for matmul then launches SEPARATE
   Triton kernels for bias/activation/normalization. A fused Triton epilogue
   kernel reads the matmul output once and applies all post-ops in one pass,
   eliminating intermediate global memory writes. This is the #1 opportunity.
2. CROSS-REDUCTION FUSION: Inductor keeps reductions and subsequent elementwise
   ops as separate kernels. Example: LayerNorm decomposes into mean kernel +
   variance kernel + normalize kernel (3 separate launches). A fused Triton
   version reads data once, computes mean+var+normalize in a single pass.
3. MULTI-OP SEQUENCES: When the computation has op1 → op2 → op3 where each op
   touches the same data, Inductor generates 3 separate kernels with
   intermediate buffers. A fused Triton kernel processes the entire sequence
   in one pass, keeping intermediate values in registers/SRAM.
4. CUSTOM ALGORITHMS: Inductor cannot discover algorithmic improvements.
   Example: standard attention is O(N^2) memory; flash attention is O(N).
   Inductor generates the naive multi-kernel version; Triton can implement
   the optimal online algorithm with tiled accumulation.
5. SPECIALIZED TILING: Inductor uses generic heuristic-based tiling. For
   specific tensor shapes common in LLMs (hidden_dim=4096, seq_len=2048),
   hand-tuned tile sizes via @triton.autotune consistently outperform
   Inductor's generic choices.

When analyzing the ATen ops, ALWAYS reason about what Inductor would do:
- Does it use cuBLAS/cuDNN for matmul/conv? (YES for standard dense ops)
- How many epilogue kernels does it launch AFTER the library call?
- Can you fuse those epilogue kernels into one?
- Is there an algorithmic improvement possible (not just tiling)?

FUSION PATTERNS — detect and fuse these when the ATen ops form a sequence:
1. NORMALIZATION: Single-pass LayerNorm/RMSNorm that computes
   mean+var+normalize+scale in one kernel. Do NOT decompose into 2+ passes.
2. ACTIVATION + BIAS: Fuse bias_add + GELU/SiLU/ReLU into one kernel.
   Load once, compute both, store once.
3. RESIDUAL + NORM: Fuse x = residual + x; x = LayerNorm(x) into one kernel.
4. GATED ACTIVATIONS: Fuse SwiGLU (gate * silu(up)) into one kernel.
5. MATMUL EPILOGUE: Fuse matmul + bias + activation using tl.dot.
6. CONV + BN + RELU: Fuse convolution output + batch normalization + activation.
7. GEMM + SwiGLU: For y = silu(x @ W_gate) * (x @ W_up), concatenate W_gate
   and W_up into B(K, 2N), perform one GEMM A(M,K) x B(K,2N) -> C(M,2N), split
   output in epilogue: gate=C[:,:N], up=C[:,N:], result = silu(gate) * up.
8. GEMM + AMAX: After GEMM accumulation, compute per-row/column absolute maximum
   for quantization calibration inline in the epilogue. No separate reduction kernel.
9. MoE ROUTING: For Mixture of Experts, use tl.program_id to route tokens to
   the correct expert weights via per-expert pointer arrays.

ANTI-CHEATING CONSTRAINTS:

All core computation logic MUST be implemented in @triton.jit kernels.
The wrapper function (kernel_function) may ONLY allocate tensors, check
dtypes/device, and launch the kernel. No math in the wrapper. PyTorch is
allowed only for allocation, dtype/device checks, and packaging results.

A kernel is considered "cheating" if it delegates the actual computation to
PyTorch instead of implementing it in Triton. The following are ALL banned:

  Banned matrix/linear algebra delegation:
  - torch.matmul / torch.mm / torch.bmm / torch.addmm
  - torch.einsum / @ operator (tensor matmul)
  - F.linear (delegates to torch.mm internally)

  Banned convolution delegation:
  - F.conv1d / F.conv2d / F.conv3d
  - torch.conv1d / torch.conv2d / torch.conv3d
  - nn.Conv1d / nn.Conv2d / nn.Conv3d

  Banned normalization delegation:
  - F.layer_norm / F.group_norm / F.batch_norm / F.instance_norm
  - nn.LayerNorm / nn.GroupNorm / nn.BatchNorm*d

  Banned activation delegation:
  - F.gelu / F.silu / F.relu / F.sigmoid / F.tanh / F.softmax / F.log_softmax
  - F.scaled_dot_product_attention

  Banned loss/other delegation:
  - F.cross_entropy / F.nll_loss / F.mse_loss
  - F.dropout
  - nn.Module computation in forward()

  Banned low-level delegation:
  - extern_kernels.* (Inductor cuBLAS/cuDNN delegation)
  - torch.ops.aten.* low-level operator calls
  - Any torch.nn.* module that performs the actual computation

  Also banned:
  - Trivial identity or no-op computation (copy input to output, add 0, multiply by 1)
  - Omitting necessary computations (e.g., LayerNorm without mean/variance,
    Softmax without max subtraction, Convolution without sliding window dot product)

The kernel must implement the computation using Triton primitives:
- Use tl.program_id() for work distribution across blocks
- Use tl.arange() for vectorized index generation within blocks
- Use appropriate BLOCK_SIZE parameters (powers of 2: 64, 128, 256, 512, 1024)
- Use tl.dot for matrix operations with proper accumulator
- Use tl.load/tl.store with proper masking for boundary safety
- Use tl.reduce for reductions with proper axis

TRITON KERNEL STRUCTURE GUIDELINES:

1. KERNEL STRUCTURE:
   - Use @triton.jit decorator for kernel functions
   - Use tl.constexpr for compile-time constants (BLOCK_SIZE, etc.)
   - Use @triton.autotune with at least 4 configs varying tile sizes and num_warps

2. MEMORY ACCESS PATTERNS:
   - Coalesce memory accesses: adjacent threads access adjacent memory locations
   - Use tl.load and tl.store with proper masking for out-of-bounds protection
   - Use cache_modifier='.cg' or eviction_policy='evict_last' for data not reused
   - For multi-dimensional tensors, compute linear offsets from strides

3. NUMERICAL STABILITY:
   - Accumulate reductions and tl.dot results in float32:
       acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
       acc = tl.dot(a, b, acc=acc)
   - For LayerNorm/RMSNorm, accumulate mean and variance in fp32:
       mean_f32 = tl.zeros((), dtype=tl.float32)
       var_f32  = tl.zeros((), dtype=tl.float32)
   - Cast to output dtype only on the final tl.store

4. COMMON PATTERNS WITH CODE:
   a) Fused BatchNorm inside convolution (after tl.dot accumulation):
       bn_mean = tl.load(mean_ptr + oc).to(tl.float32)
       bn_var = tl.load(var_ptr + oc).to(tl.float32)
       gamma = tl.load(gamma_ptr + oc).to(tl.float32)
       beta = tl.load(beta_ptr + oc).to(tl.float32)
       x_norm = (acc.to(tl.float32) - bn_mean) / tl.sqrt(bn_var + eps)
       out = x_norm * gamma + beta
       tl.store(out_ptr + idx, out.to(out_ptr.dtype.element_ty), mask=mask)

   b) Implicit GEMM Convolution (Conv2d forward, adapts to Conv3d/ConvTranspose):
       # Convolution = GEMM where:
       #   M = Batch * OutH * OutW  (output spatial positions)
       #   N = OutChannels
       #   K = InChannels * KernelH * KernelW  (reduction over filter volume)
       # Input X: (N, Ci, H, W) in NCHW.  Weight W: (Co, Ci, R, S).
       # Compute spatial offsets on-the-fly inside the K loop (no im2col copy).
       #
       # KERNEL (use channels-last NHWC internally for coalesced access):
       pid_m = tl.program_id(0)   # output spatial tile
       pid_n = tl.program_id(1)   # output channel tile
       offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
       offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
       # Decode linear M-offset → (batch, out_h, out_w)
       out_w = offs_m % OUT_W
       out_h = (offs_m // OUT_W) % OUT_H
       batch = offs_m // (OUT_W * OUT_H)
       acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
       TOTAL_K = IN_C * KERNEL_H * KERNEL_W
       for k_off in range(0, TOTAL_K, BLOCK_K):
           offs_k = k_off + tl.arange(0, BLOCK_K)
           # Decode K-offset → (in_c, kh, kw)
           in_c = offs_k % IN_C
           kw   = (offs_k // IN_C) % KERNEL_W
           kh   = offs_k // (IN_C * KERNEL_W)
           # Map to input coordinates with stride and padding
           ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
           iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W
           valid = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (offs_k[None, :] < TOTAL_K)
           # Load input tile [BLOCK_M, BLOCK_K]
           x_ptrs = X_ptr + batch[:, None]*s_xn + in_c[None,:]*s_xc + ih*s_xh + iw*s_xw
           x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
           # Load weight tile [BLOCK_K, BLOCK_N]
           w_ptrs = W_ptr + offs_n[None,:]*s_wo + in_c[:,None]*s_wi + kh[:,None]*s_wkh + kw[:,None]*s_wkw
           w_tile = tl.load(w_ptrs, mask=(offs_k[:,None] < TOTAL_K) & (offs_n[None,:] < OUT_C), other=0.0)
           acc = tl.dot(x_tile, w_tile, acc)
       # Store output
       y_ptrs = Y_ptr + batch[:,None]*s_yn + out_h[:,None]*s_yh + out_w[:,None]*s_yw + offs_n[None,:]*s_yc
       tl.store(y_ptrs, acc, mask=(offs_m[:,None] < M_TOTAL) & (offs_n[None,:] < OUT_C))
       #
       # WRAPPER:
       # def kernel_function(x, w):  # x:(N,Ci,H,W), w:(Co,Ci,R,S)
       #     x_nhwc = x.permute(0,2,3,1).contiguous()  # NCHW → NHWC for coalescing
       #     w_flat = w.reshape(Co, Ci*R*S)              # flatten K dim
       #     OUT_H = (H + 2*pad - R) // stride + 1
       #     OUT_W = (W + 2*pad - S) // stride + 1
       #     M = N * OUT_H * OUT_W
       #     y = torch.empty(M, Co, device=x.device, dtype=x.dtype)
       #     grid = (cdiv(M, BLOCK_M), cdiv(Co, BLOCK_N))
       #     conv_kernel[grid](x_nhwc, w, y, ...)
       #     return y.reshape(N, OUT_H, OUT_W, Co).permute(0,3,1,2)  # back to NCHW
       #
       # FOR CONV3D: add out_d coordinate in M-decode, kz in K-decode, id in input mapping.
       #   M = Batch * OutD * OutH * OutW;  K = InC * KD * KH * KW
       #   id = out_d[:, None] * STRIDE_D + kz[None, :] - PAD_D
       #
       # FOR CONVTRANSPOSE: swap gather→scatter. For each INPUT position, scatter
       #   to multiple output positions. Use tl.atomic_add for overlapping writes.
       #   Output size = (H - 1) * stride - 2*pad + dilation*(R-1) + output_padding + 1
       #
       # FOR DILATION: replace ih = out_h*S + kh*D - P (multiply kernel offset by dilation).
       # FOR GROUPS: partition channels: group_ci = Ci//groups, loop over group_ci only.

   c) Online softmax (numerically stable, single-pass):
       row_max = tl.full((), float('-inf'), dtype=tl.float32)
       row_sum = tl.zeros((), dtype=tl.float32)
       # Pass 1: find max and accumulate shifted exp sum
       for off in range(0, N, BLOCK):
           x = tl.load(input_ptr + off + tl.arange(0, BLOCK), mask=...)
           new_max = tl.maximum(row_max, tl.max(x, axis=0))
           row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(x - new_max))
           row_max = new_max
       # Pass 2: normalize
       for off in range(0, N, BLOCK):
           x = tl.load(input_ptr + off + tl.arange(0, BLOCK), mask=...)
           out = tl.exp(x - row_max) / row_sum
           tl.store(output_ptr + off + tl.arange(0, BLOCK), out, mask=...)

5. FUSION PRIORITY:
   - Inspect the ATen ops to identify sequences that can run in a single pass
   - Prefer a single fused Triton kernel that covers the entire operator pipeline
     (e.g., combine matmul + bias + activation, or mean + variance + normalize)
   - Only fall back to unfused implementation if fusion is provably impossible
   - Fusing saves kernel launch overhead AND eliminates intermediate buffer writes

OUTPUT FORMAT — return exactly this structure:
```
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[...],
    key=[...],
)
@triton.jit
def kernel_name(...):
    ...

def kernel_function(*inputs):
    ...
    return output

def get_inputs():
    return [...]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    if not args.profile_only:
        # Verify correctness against PyTorch reference using torch.allclose
        # (NEVER use assert max_diff < threshold — use relative tolerance)
        ref = <compute reference using PyTorch eager ops>
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \\
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")
```

FEW-SHOT EXAMPLE — Fused Softmax:

PyTorch (naive — 5 separate ops, 5MN+2M reads, 3MN+2M writes):
```python
def naive_softmax(x):
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret
```

Optimized Triton (single-pass, reads MN once, writes MN once — ~4x less traffic):
```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```
Key insight: Fuses max, subtract, exp, sum, divide into one kernel. Each row
is loaded once into SRAM. No intermediate buffers needed.
"""


def generate_triton_kernel(
    kernel_group: dict,
    llm_config: dict,
    gpu_specs: dict | None = None,
) -> str | None:
    """Generate a clean Triton kernel for a kernel group.

    Args:
        kernel_group: Dict from graph_tracer.trace_model() with:
            - kernel_name, kernel_type
            - fused_aten_ops, aten_graph
            - model_inputs (shapes/dtypes)
            - inductor_source (optional reference)
        llm_config: Dict with model, api_key, project_id, region.
        gpu_specs: Optional GPU hardware info.

    Returns:
        Complete Python source for the standalone kernel, or None on failure.
    """
    try:
        llm = build_llm(
            model=llm_config.get("model", "claude-sonnet-4-6"),
            api_key=llm_config.get("api_key"),
            max_tokens=8192,
            temperature=0.1,
            project_id=llm_config.get("project_id"),
            region=llm_config.get("region"),
            provider=llm_config.get("provider"),
            openai_base_url=llm_config.get("openai_base_url"),
            hf_inference_provider=llm_config.get("hf_inference_provider", "auto"),
        )
    except Exception as exc:
        pipe_print(f"[codegen] Failed to build LLM: {exc}")
        return None

    prompt = _build_codegen_prompt(kernel_group, gpu_specs)

    try:
        response = llm.invoke([
            SystemMessage(content=CODEGEN_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        from kernelopt.llm import strip_thinking
        source = strip_thinking(response.content) if response.content else ""
        if not source or not source.strip():
            pipe_print("[codegen] LLM returned empty response.")
            return None

        source = _sanitize_output(source)
        pipe_print(f"[codegen] Generated kernel ({len(source)} chars)")
        return source
    except Exception as exc:
        pipe_print(f"[codegen] LLM invocation failed: {exc}")
        return None


def _build_codegen_prompt(kernel_group: dict, gpu_specs: dict | None) -> str:
    """Build the human message describing what kernel to write."""
    kg = kernel_group
    lines = []

    lines.append("## Kernel Specification")
    lines.append("")

    # ATen ops
    aten_ops = kg.get("fused_aten_ops", [])
    aten_graph = kg.get("aten_graph", [])
    if aten_ops:
        lines.append(f"**Operations to implement:** {', '.join(aten_ops)}")
    elif aten_graph:
        op_names = [op["name"] for op in aten_graph]
        lines.append(f"**Operations to implement:** {', '.join(op_names)}")

    # Kernel type hint
    ktype = kg.get("kernel_type", "unknown")
    if ktype == "pointwise":
        lines.append("**Kernel type:** Pointwise (element-wise operation)")
    elif ktype == "reduction":
        lines.append("**Kernel type:** Reduction (reduces along one or more dimensions)")
    elif ktype == "template":
        lines.append("**Kernel type:** GEMM/Matmul (tiled matrix multiplication with tl.dot)")
    elif ktype == "extern":
        lines.append("**Kernel type:** Library-backed operation (cuBLAS/cuDNN)")
    elif ktype == "from_aten":
        lines.append("**Kernel type:** Derive from the ATen op specification")
    lines.append("")

    # Inductor extern_ops context — tell LLM which ops use library calls
    extern_ops = kg.get("extern_ops", [])
    if extern_ops:
        lines.append("**INDUCTOR ANALYSIS — Library Calls Detected:**")
        lines.append("Inductor (torch.compile) uses cuBLAS/cuDNN for the following operations:")
        for eop in extern_ops:
            lines.append(f"  - `{eop['raw_call'][:120]}`")
            lines.append(f"    Reason: {eop['reason']}")
        inductor_pattern = kg.get("inductor_pattern", "")
        if inductor_pattern:
            lines.append(f"  Inductor pattern: {inductor_pattern}")
        lines.append("")
        lines.append("**YOUR STRATEGY:**")
        lines.append("1. Keep the library call as a PyTorch op (torch.mm, torch.addmm, F.conv2d, etc.)")
        lines.append("2. Write a @triton.jit kernel ONLY for the epilogue ops (activation, norm, scaling)")
        lines.append("3. In kernel_function: call the PyTorch op, then pass output to your Triton epilogue")
        lines.append("4. Only replace the library call with Triton if the ORIGINAL PyTorch source reveals")
        lines.append("   the computation can be done with a fundamentally different algorithm that avoids")
        lines.append("   the full dense matmul/conv (e.g., the operation has exploitable structure like")
        lines.append("   sparsity, diagonal structure, or can be fused into an online algorithm).")
        lines.append("   Examine the original forward() method to determine this — do NOT just rewrite")
        lines.append("   a standard dense torch.mm as tl.dot, that is always slower than cuBLAS.")
        lines.append("")

    # Input shapes
    model_inputs = kg.get("model_inputs", [])
    if model_inputs:
        lines.append("**Model inputs:**")
        for inp in model_inputs:
            lines.append(f"  - {inp['name']}: shape={inp['shape']}, dtype={inp['dtype']}")
        lines.append("")

    # ATen graph details (shapes of intermediate ops)
    if aten_graph:
        lines.append("**Computation graph (ATen ops):**")
        for op in aten_graph:
            shape_str = f" → shape={op['shape']}" if op.get("shape") else ""
            dtype_str = f" dtype={op['dtype']}" if op.get("dtype") and op["dtype"] != "?" else ""
            lines.append(f"  - {op['name']}{shape_str}{dtype_str}")
        lines.append("")

    # GPU specs
    if gpu_specs:
        lines.append("**GPU:**")
        lines.append(f"  - {gpu_specs.get('gpu_name', 'unknown')}")
        lines.append(f"  - {gpu_specs.get('sm_count', '?')} SMs")
        lines.append(f"  - Compute capability: {gpu_specs.get('compute_capability', '?')}")
        lines.append("")

    # Original PyTorch source (if available) — shows exact math/algorithm
    pytorch_src = kg.get("pytorch_source", "")
    if pytorch_src and len(pytorch_src) > 50:
        lines.append("**Original PyTorch implementation (implement the EXACT same math):**")
        lines.append("```python")
        lines.append(pytorch_src[:MAX_SOURCE_CHARS])
        lines.append("```")
        lines.append("")

    # Full Inductor output — gives LLM complete decomposition context
    full_inductor = kg.get("full_inductor_source", "") or kg.get("inductor_source", "")
    if full_inductor and len(full_inductor) > 100:
        lines.append(
            "**Inductor's complete generated code for this model** "
            "(shows every kernel and how they are called in `call()` — "
            "understand the full decomposition, then write a single optimized kernel):"
        )
        lines.append("```python")
        lines.append(full_inductor[:24_000])
        lines.append("```")
        lines.append("")

    # Compile error context (if force_triton failed)
    if kg.get("compile_error"):
        lines.append("**Note:** torch.compile could not generate a Triton kernel for this op.")
        lines.append("You must write the Triton kernel from scratch based on the ATen op spec.")

    # Previous attempt error (retry feedback)
    if kg.get("_codegen_error"):
        lines.append("")
        lines.append("**PREVIOUS ATTEMPT FAILED — fix the error below:**")
        lines.append(f"```\n{kg['_codegen_error'][-1000:]}\n```")
        lines.append("Diagnose what went wrong, fix it, and produce a corrected kernel.")
        lines.append("")

    lines.append("Write a complete, self-contained Python file with the Triton kernel.")

    return "\n".join(lines)


def _sanitize_output(source: str) -> str:
    """Clean LLM output: extract code block, strip preamble, fix Unicode."""
    import re

    from kernelopt.fusion.generate import sanitize_llm_output

    # Extract python code block if wrapped in markdown
    m = re.search(r'```python\n(.*?)```', source, re.DOTALL)
    if m:
        source = m.group(1)
    elif source.startswith("```"):
        source = re.sub(r'^```\w*\n', '', source)
        source = re.sub(r'\n```\s*$', '', source)
    else:
        # Strip text preamble
        lines = source.split('\n')
        code_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and (
                stripped.startswith('import ')
                or stripped.startswith('from ')
                or stripped.startswith('@')
                or stripped.startswith('def ')
                or stripped.startswith('#!')
                or stripped.startswith('"""')
            ):
                code_start = i
                break
        if code_start > 0:
            source = '\n'.join(lines[code_start:])

    # Apply Unicode/prose sanitization (fixes ×→*, →→->, strips remaining prose)
    source = sanitize_llm_output(source)

    return source.strip()
