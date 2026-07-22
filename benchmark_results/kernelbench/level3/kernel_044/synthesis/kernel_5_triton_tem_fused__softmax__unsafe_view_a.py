import torch
import triton
import triton.language as tl
import math

# ─── Fused LayerNorm kernel ───────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=4),
        triton.Config({'BLOCK_N': 512},  num_warps=8),
        triton.Config({'BLOCK_N': 512},  num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * rstd

    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w + b
    tl.store(out_ptr + row * N + cols, out, mask=mask)


# ─── Fused QKV bias + reshape kernel ─────────────────────────────────────────
# Adds bias to QKV matmul output and reshapes for multi-head attention
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def add_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    numel, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    bias_idx = offs % N
    b = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + b, mask=mask)


# ─── Fused attention score kernel: QK^T * scale + mask + softmax ─────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_T': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 64},  num_warps=8, num_stages=2),
    ],
    key=['T'],
)
@triton.jit
def attn_softmax_kernel(
    # att: [B, nh, T, T] (input/output)
    att_ptr,
    B, nh, T,
    BLOCK_T: tl.constexpr,
):
    # Each program handles one row of the attention matrix
    pid = tl.program_id(0)
    total_rows = B * nh * T
    if pid >= total_rows:
        return

    b_idx = pid // (nh * T)
    nh_idx = (pid // T) % nh
    t_idx = pid % T

    row_ptr = att_ptr + (b_idx * nh * T * T + nh_idx * T * T + t_idx * T)
    bias_row_ptr = bias_ptr + t_idx * T  # bias is [1,1,T,T]

    # Load row and apply causal mask
    cols = tl.arange(0, BLOCK_T)
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros((), dtype=tl.float32)

    # Two-pass softmax with masking
    # First pass: find max
    for start in tl.range(0, T, BLOCK_T):
        c = start + cols
        mask = c < T
        att_val = tl.load(row_ptr + c, mask=mask, other=float('-inf')).to(tl.float32)
        bias_val = tl.load(bias_row_ptr + c, mask=mask, other=0.0).to(tl.float32)
        # Apply causal mask: where bias==0, set to -inf
        att_val = tl.where(bias_val == 0.0, float('-inf'), att_val)
        cur_max = tl.max(att_val, axis=0)
        row_max = tl.maximum(row_max, cur_max)

    # Second pass: compute sum of exp
    for start in tl.range(0, T, BLOCK_T):
        c = start + cols
        mask = c < T
        att_val = tl.load(row_ptr + c, mask=mask, other=float('-inf')).to(tl.float32)
        bias_val = tl.load(bias_row_ptr + c, mask=mask, other=0.0).to(tl.float32)
        att_val = tl.where(bias_val == 0.0, float('-inf'), att_val)
        exp_val = tl.exp(att_val - row_max)
        exp_val = tl.where(mask, exp_val, 0.0)
        row_sum += tl.sum(exp_val, axis=0)

    # Third pass: normalize and store
    for start in tl.range(0, T, BLOCK_T):
        c = start + cols
        mask = c < T
        att_val = tl.load(row_ptr + c, mask=mask, other=float('-inf')).to(tl.float32)
        bias_val = tl.load(bias_row_ptr + c, mask=mask, other=0.0).to(tl.float32)
        att_val = tl.where(bias_val == 0.0, float('-inf'), att_val)
        softmax_val = tl.exp(att_val - row_max) / row_sum
        softmax_val = tl.where(mask, softmax_val, 0.0)
        tl.store(row_ptr + c, softmax_val.to(tl.float32), mask=mask)


# ─── Fused GELU kernel ────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def gelu_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # NewGELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # math.sqrt(2.0 / math.pi)
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + triton.language.extra.libdevice.tanh(inner))
    tl.store(out_ptr + offs, gelu, mask=mask)


# ─── Fused linear + bias + add (residual) kernel ─────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def bias_add_residual_kernel(
    x_ptr, bias_ptr, residual_ptr, out_ptr,
    numel, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    bias_idx = offs % N
    b = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + b + res, mask=mask)


# ─── Tiled GEMM kernel ────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=c_mask)


def triton_matmul(A, B):
    """A: [M, K], B: [K, N] -> C: [M, N]"""
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    K2, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


# ─── Batched matmul for attention: [B*nh, T, hs] x [B*nh, hs, T] -> [B*nh, T, T] ─
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K', 'batch'],
)
@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    batch,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(1)
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


def triton_batched_matmul(A, B):
    """A: [batch, M, K], B: [batch, K, N] -> C: [batch, M, N]"""
    batch, M, K = A.shape
    _, K2, N = B.shape
    C = torch.empty((batch, M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), batch)
    batched_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        batch,
    )
    return C


def kernel_function(
    p_ln_1_weight, p_ln_1_bias,
    p_attn_c_attn_weight, p_attn_c_attn_bias,
    p_attn_c_proj_weight, p_attn_c_proj_bias,
    p_ln_2_weight, p_ln_2_bias,
    p_mlp_c_fc_weight, p_mlp_c_fc_bias,
    p_mlp_c_proj_weight, p_mlp_c_proj_bias,
    b_attn_bias,
    x,
):
    B, T, C = x.shape  # 128, 512, 768
    n_head = 8
    hs = C // n_head  # 96
    eps = 1e-5

    # ── 1. LayerNorm 1 ────────────────────────────────────────────────────────
    ln1_out = torch.empty_like(x)
    M_ln = B * T
    N_ln = C
    grid_ln = (M_ln,)
    layer_norm_kernel[grid_ln](
        x, p_ln_1_weight, p_ln_1_bias, ln1_out,
        M_ln, N_ln, eps,
    )

    # ── 2. QKV projection: [B*T, C] x [C, 3C]^T -> [B*T, 3C] ────────────────
    ln1_2d = ln1_out.view(B * T, C)
    # p_attn_c_attn_weight is [2304, 768], so W^T is [768, 2304]
    W_qkv = p_attn_c_attn_weight  # [2304, 768]
    qkv_mm = triton_matmul(ln1_2d, W_qkv.t().contiguous())  # [B*T, 2304]

    # Add bias
    qkv = torch.empty_like(qkv_mm)
    numel_qkv = qkv_mm.numel()
    grid_bias = lambda meta: (triton.cdiv(numel_qkv, meta['BLOCK_SIZE']),)
    add_bias_kernel[grid_bias](
        qkv_mm, p_attn_c_attn_bias, qkv,
        numel_qkv, 2304,
    )
    qkv = qkv.view(B, T, 3 * C)

    # ── 3. Split Q, K, V ──────────────────────────────────────────────────────
    q = qkv[:, :, :C].contiguous()        # [B, T, C]
    k = qkv[:, :, C:2*C].contiguous()     # [B, T, C]
    v = qkv[:, :, 2*C:].contiguous()      # [B, T, C]

    # Reshape to multi-head: [B, nh, T, hs]
    q = q.view(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()  # [B, nh, T, hs]
    k = k.view(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()  # [B, nh, T, hs]
    v = v.view(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()  # [B, nh, T, hs]

    # ── 4. Attention scores: Q @ K^T * scale ─────────────────────────────────
    scale = 1.0 / math.sqrt(hs)
    # [B, nh, T, hs] x [B, nh, hs, T] -> [B, nh, T, T]
    q_2d = q.view(B * n_head, T, hs)
    k_t = k.permute(0, 1, 3, 2).contiguous().view(B * n_head, hs, T)
    att = triton_batched_matmul(q_2d, k_t)  # [B*nh, T, T]
    att = att.view(B, n_head, T, T) * scale

    # ── 5. Causal mask + softmax ──────────────────────────────────────────────
    # b_attn_bias: [1, 1, 1024, 1024], we need [:, :, :T, :T]
    causal_bias = b_attn_bias[0, 0, :T, :T].contiguous()  # [T, T]

    # Apply mask and softmax in-place
    total_rows = B * n_head * T
    grid_softmax = (total_rows,)
    attn_softmax_kernel[grid_softmax](
        att, causal_bias,
        B, n_head, T,
    )
    # att is now softmax output [B, nh, T, T]
    # dropout is identity at inference

    # ── 6. att @ V ────────────────────────────────────────────────────────────
    att_2d = att.view(B * n_head, T, T)
    v_2d = v.view(B * n_head, T, hs)
    y = triton_batched_matmul(att_2d, v_2d)  # [B*nh, T, hs]
    y = y.view(B, n_head, T, hs).permute(0, 2, 1, 3).contiguous().view(B, T, C)

    # ── 7. Output projection + residual ──────────────────────────────────────
    y_2d = y.view(B * T, C)
    proj_mm = triton_matmul(y_2d, p_attn_c_proj_weight.t().contiguous())  # [B*T, C]
    # Add bias + residual
    attn_out = torch.empty(B * T, C, device=x.device, dtype=x.dtype)
    numel_proj = proj_mm.numel()
    grid_res = lambda meta: (triton.cdiv(numel_proj, meta['BLOCK_SIZE']),)
    bias_add_residual_kernel[grid_res](
        proj_mm, p_attn_c_proj_bias, x.view(B * T, C), attn_out,
        numel_proj, C,
    )
    attn_out = attn_out.view(B, T, C)

    # ── 8. LayerNorm 2 ────────────────────────────────────────────────────────
    ln2_out = torch.empty_like(attn_out)
    layer_norm_kernel[grid_ln](
        attn_out, p_ln_2_weight, p_ln_2_bias, ln2_out,
        M_ln, N_ln, eps,
    )

    # ── 9. MLP FC: [B*T, C] x [C, 4C]^T -> [B*T, 4C] ────────────────────────
    ln2_2d = ln2_out.view(B * T, C)
    fc_mm = triton_matmul(ln2_2d, p_mlp_c_fc_weight.t().contiguous())  # [B*T, 3072]

    # Add bias
    fc_out = torch.empty_like(fc_mm)
    numel_fc = fc_mm.numel()
    grid_fc = lambda meta: (triton.cdiv(numel_fc, meta['BLOCK_SIZE']),)
    add_bias_kernel[grid_fc](
        fc_mm, p_mlp_c_fc_bias, fc_out,
        numel_fc, 3072,
    )

    # ── 10. NewGELU activation ────────────────────────────────────────────────
    gelu_out = torch.empty_like(fc_out)
    grid_gelu = lambda meta: (triton.cdiv(numel_fc, meta['BLOCK_SIZE']),)
    gelu_kernel[grid_gelu](
        fc_out, gelu_out,
        numel_fc,
    )

    # ── 11. MLP proj: [B*T, 4C] x [4C, C]^T -> [B*T, C] ─────────────────────
    gelu_2d = gelu_out.view(B * T, 3072)
    proj2_mm = triton_matmul(gelu_2d, p_mlp_c_proj_weight.t().contiguous())  # [B*T, C]

    # Add bias + residual
    mlp_out = torch.empty(B * T, C, device=x.device, dtype=x.dtype)
    numel_mlp = proj2_mm.numel()
    grid_mlp = lambda meta: (triton.cdiv(numel_mlp, meta['BLOCK_SIZE']),)
    bias_add_residual_kernel[grid_mlp](
        proj2_mm, p_mlp_c_proj_bias, attn_out.view(B * T, C), mlp_out,
        numel_mlp, C,
    )
    mlp_out = mlp_out.view(B, T, C)

    return mlp_out


def get_inputs():
    torch.manual_seed(42)
    B, T, C = 128, 512, 768
    n_head = 8
    max_seqlen = 1024

    p_ln_1_weight = torch.randn(C, device='cuda')
    p_ln_1_bias = torch.randn(C, device='cuda')