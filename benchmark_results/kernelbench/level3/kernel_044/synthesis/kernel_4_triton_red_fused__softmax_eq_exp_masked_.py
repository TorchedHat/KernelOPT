import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused LayerNorm kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 768}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    out = diff * rstd * w + b

    tl.store(out_ptr + row * N + cols, out.to(tl.float32), mask=mask)


# ============================================================
# GEMM kernel (M x K) x (K x N) -> (M x N)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


def gemm(a, b, bias=None):
    """Compute a @ b.T + bias where b is (out_features, in_features)"""
    assert a.ndim == 2 and b.ndim == 2
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_kernel[grid](
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(1), b.stride(0),  # b is transposed: stride_bk=b.stride(1), stride_bn=b.stride(0)
        c.stride(0), c.stride(1),
        HAS_BIAS=(bias is not None),
    )
    return c


# ============================================================
# Fused attention: QK^T * scale + mask, softmax
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64}, num_warps=4),
        triton.Config({'BLOCK_T': 128}, num_warps=8),
        triton.Config({'BLOCK_T': 256}, num_warps=8),
        triton.Config({'BLOCK_T': 512}, num_warps=16),
    ],
    key=['T'],
)
@triton.jit
def attn_softmax_kernel(
    att_ptr, bias_ptr, out_ptr,
    B, NH, T,
    BLOCK_T: tl.constexpr,
):
    # Each program handles one row of the attention matrix
    # program_id(0) = batch * n_heads * T + head * T + t
    row_idx = tl.program_id(0)
    b = row_idx // (NH * T)
    remainder = row_idx % (NH * T)
    h = remainder // T
    t = remainder % T

    cols = tl.arange(0, BLOCK_T)
    mask = cols < T

    # Load attention scores for this row
    att = tl.load(att_ptr + row_idx * T + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # Load causal mask bias (1, 1, T, T) -> bias[0, 0, t, :]
    # bias_ptr points to (1, 1, 1024, 1024), we need [0, 0, t, :T]
    bias_val = tl.load(bias_ptr + t * 1024 + cols, mask=mask, other=0.0).to(tl.float32)

    # Apply causal mask: where bias == 0, set to -inf
    att = tl.where(bias_val == 0.0, float('-inf'), att)

    # Softmax
    att_max = tl.max(att, axis=0)
    att = att - att_max
    att_exp = tl.exp(att)
    att_sum = tl.sum(att_exp, axis=0)
    att_softmax = att_exp / att_sum

    tl.store(out_ptr + row_idx * T + cols, att_softmax.to(tl.float32), mask=mask)


# ============================================================
# Batched matmul for attention: (B, NH, T, T) x (B, NH, T, HS) -> (B, NH, T, HS)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['T', 'HS'],
)
@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    B, NH, T, HS,
    stride_ab, stride_ah, stride_am, stride_ak,
    stride_bb, stride_bh, stride_bk, stride_bn,
    stride_cb, stride_ch, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program_id(0) = tile over M (T dimension)
    # program_id(1) = tile over N (HS dimension)
    # program_id(2) = batch * NH
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    b_idx = pid_bh // NH
    h_idx = pid_bh % NH

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = a_ptr + b_idx * stride_ab + h_idx * stride_ah
    b_base = b_ptr + b_idx * stride_bb + h_idx * stride_bh

    a_ptrs = a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(T, BLOCK_K)):
        k_mask = offs_k < T - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < T) & (k_mask[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (offs_n[None, :] < HS), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_base = c_ptr + b_idx * stride_cb + h_idx * stride_ch
    c_ptrs = c_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < T) & (offs_n[None, :] < HS)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


# ============================================================
# Batched matmul for QK^T: (B, NH, T, HS) x (B, NH, HS, T) -> (B, NH, T, T)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['T', 'HS'],
)
@triton.jit
def qk_matmul_kernel(
    q_ptr, k_ptr, out_ptr,
    B, NH, T, HS,
    scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kk, stride_kn,
    stride_ob, stride_oh, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    b_idx = pid_bh // NH
    h_idx = pid_bh % NH

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q_base = q_ptr + b_idx * stride_qb + h_idx * stride_qh
    k_base = k_ptr + b_idx * stride_kb + h_idx * stride_kh

    # Q: (T, HS), K^T: (HS, T) -> out: (T, T)
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = k_base + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(HS, BLOCK_K)):
        k_mask = offs_k < HS - k * BLOCK_K
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < T) & (k_mask[None, :]), other=0.0)
        k_tile = tl.load(k_ptrs, mask=(k_mask[:, None]) & (offs_n[None, :] < T), other=0.0)
        acc = tl.dot(q, k_tile, acc, allow_tf32=False)
        q_ptrs += BLOCK_K * stride_qk
        k_ptrs += BLOCK_K * stride_kk

    acc = acc * scale

    out_base = out_ptr + b_idx * stride_ob + h_idx * stride_oh
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < T) & (offs_n[None, :] < T)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


# ============================================================
# Fused GELU kernel: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(
    x_ptr, out_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    tanh_val = tl.extra.cuda.libdevice.tanh(inner)
    out = 0.5 * x * (1.0 + tanh_val)

    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


# ============================================================
# Main kernel function
# ============================================================
def kernel_function(
    p_ln_1_weight, p_ln_1_bias,
    p_attn_c_attn_weight, p_attn_c_attn_bias,
    p_attn_c_proj_weight, p_attn_c_proj_bias,
    p_ln_2_weight, p_ln_2_bias,
    p_mlpf_c_fc_weight, p_mlpf_c_fc_bias,
    p_mlpf_c_proj_weight, p_mlpf_c_proj_bias,
    b_attn_bias, x,
):
    B, T, C = x.shape  # 128, 512, 768
    n_head = 8
    hs = C // n_head  # 96
    NH = n_head

    # ---- LayerNorm 1 ----
    ln1_out = torch.empty_like(x)
    grid_ln = (B * T,)
    layernorm_kernel[grid_ln](
        x, p_ln_1_weight, p_ln_1_bias, ln1_out,
        B * T, C, 1e-5,
    )

    # ---- Linear: QKV projection ----
    # ln1_out: (B*T, C) -> qkv: (B*T, 3*C)
    ln1_flat = ln1_out.view(B * T, C)
    qkv = gemm(ln1_flat, p_attn_c_attn_weight, p_attn_c_attn_bias)
    qkv = qkv.view(B, T, 3 * C)

    # Split Q, K, V
    q = qkv[:, :, :C].view(B, T, NH, hs).permute(0, 2, 1, 3).contiguous()   # (B, NH, T, HS)
    k = qkv[:, :, C:2*C].view(B, T, NH, hs).permute(0, 2, 1, 3).contiguous()  # (B, NH, T, HS)
    v = qkv[:, :, 2*C:].view(B, T, NH, hs).permute(0, 2, 1, 3).contiguous()   # (B, NH, T, HS)

    # ---- QK^T matmul ----
    scale = 1.0 / math.sqrt(hs)
    att = torch.empty(B, NH, T, T, device=x.device, dtype=x.dtype)

    grid_qk = (triton.cdiv(T, 32), triton.cdiv(T, 32), B * NH)
    qk_matmul_kernel[grid_qk](
        q, k, att,
        B, NH, T, hs,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        att.stride(0), att.stride(1), att.stride(2), att.stride(3),
    )

    # ---- Softmax with causal mask ----
    att_softmax = torch.empty_like(att)
    # bias is (1, 1, 1024, 1024), we need [0, 0, :T, :T]
    bias_slice = b_attn_bias[0, 0, :T, :T].contiguous()  # (T, T)

    grid_softmax = (B * NH * T,)
    attn_softmax_kernel[grid_softmax](
        att, bias_slice, att_softmax,
        B, NH, T,
    )

    # ---- Attention output: softmax(QK^T) @ V ----
    attn_out = torch.empty(B, NH, T, hs, device=x.device, dtype=x.dtype)
    grid_av = (triton.cdiv(T, 32), triton.cdiv(hs, 32), B * NH)
    batched_matmul_kernel[grid_av](
        att_softmax, v, attn_out,
        B, NH, T, hs,
        att_softmax.stride(0), att_softmax.stride(1), att_softmax.stride(2), att_softmax.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
    )

    # Reshape: (B, NH, T, HS) -> (B*T, C)
    y = attn_out.permute(0, 2, 1, 3).contiguous().view(B * T, C)

    # ---- Output projection ----
    proj_out = gemm(y, p_attn_c_proj_weight, p_attn_c_proj_bias)
    proj_out = proj_out.view(B, T, C)

    # ---- Residual add 1 ----
    x2 = x + proj_out  # dropout is identity at inference

    # ---- LayerNorm 2 ----
    ln2_out = torch.empty_like(x2)
    layernorm_kernel[grid_ln](
        x2, p_ln_2_weight, p_ln_2_bias, ln2_out,
        B * T, C, 1e-5,
    )

    # ---- MLP FC ----
    ln2_flat = ln2_out.view(B * T, C)
    fc_out = gemm(ln2_flat, p_mlpf_c_fc_weight, p_mlpf_c_fc_bias)  # (B*T, 3072)

    # ---- GELU ----
    gelu_out = torch.empty_like(fc_out)
    N_gelu = B * T * 3072
    grid_gelu = lambda meta: (triton.cdiv(N_gelu, meta['BLOCK_SIZE']),)
    gelu_kernel[grid_gelu](fc_out, gelu_out, N_gelu)

    # ---- MLP Proj ----
    proj2_out = gemm(gelu_out, p_mlpf_c_proj_weight, p_mlpf_c_proj_bias)  # (B*T, C)
    proj2_out = proj2_out.view(B, T, C)

    # ---- Residual add 2 ----
    out = x2 + proj2_out

    return out


def get_inputs():
    B, T, C = 128, 512, 768
    n_head = 8
    max_seqlen = 1024

    p_ln_1_weight = torch.randn(C, dtype=torch.float32, device='cuda')
    p_ln_1_bias = torch.randn(C, dtype=torch.float32, device='cuda')
    p_attn_c_attn_weight = torch.randn(3 * C, C, dtype=torch.float32, device='cuda') * 0.02
    p_attn_c_attn_bias = torch.randn(3 * C, dtype=torch.float32, device='cuda') * 0.02
    p_attn_c_proj_weight = torch.randn(C, C, dtype=torch.float32, device='cuda') * 0.02
    p_attn_c_proj_bias = torch.randn(C, dtype=torch.float32, device='cuda') * 0.02
    p_ln_2_weight = torch.randn(C, dtype=torch.float32, device='cuda')
    p_ln_2_bias = torch.randn(C, dtype=torch.float32, device='cuda')
    p_mlpf_c_fc_weight = torch.randn(3072, C, dtype=torch.float32, device='cuda') * 0.02
    p_mlpf_c_fc_bias = torch.randn(3072, dtype=torch.float32, device='cuda') * 0.02
    p_mlpf_c_proj_weight = torch.randn(C, 3072, dtype=torch.float32, device='cuda') * 0.02
    p_mlpf_c_proj_bias = torch.randn(C, dtype=torch.float32, device='cuda') * 0.02
    b_attn_bias = torch.tril(torch.ones(max_seqlen, max_seqlen, dtype=torch.float32, device='cuda')).view(1, 1, max_seqlen, max_seqlen)
    x = torch.randn(B, T, C, dtype=torch.float32, device='cuda') * 0.02

    return [
        p_ln_1_weight, p_ln_1_bias,
        p_attn_c_attn_weight, p_attn_c_attn_bias,
        p_attn_c_proj_weight, p_attn_c_proj_bias,
        p_ln_2_weight, p_ln_2_bias,
        p_mlpf_c_fc_weight, p_mlpf_c_fc_bias,
        p_mlpf_c_proj_weight, p_mlpf_c_proj_bias,
        b_attn_bias, x,
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Reference implementation
        (p_ln_1_weight, p_ln_1_bias,
         p_attn_c_attn_weight, p_attn_c_attn_bias,
         p_attn_c_proj_weight, p_attn_c_proj_bias,
         p_ln_2_weight, p_ln_2_bias,
         p_mlpf_c_fc_weight, p_mlpf_c_fc_bias,
         p_mlpf_c_proj_weight, p_mlpf_c_proj_bias,
         b_attn_bias, x) = inputs

        B, T, C = x.shape
        n_head = 8
        hs = C // n_head

        # LayerNorm 1
        import torch.nn.functional as F
        ln1 = F.layer_norm(x, [C], p_ln_1_weight, p_ln_1_bias, eps=1e-5)

        # QKV
        qkv = F.linear(ln1, p_attn_c_attn_weight, p_attn_c_attn_bias)
        q_r, k_r, v_r = qkv.split(C, dim=2)
        k_r = k_r.view(B, T, n_head, hs).transpose(1, 2)
        q_r = q_r.view(B, T, n_head, hs).transpose(1, 2)
        v_r = v_r.view(B, T, n_head, hs).transpose(1, 2)

        # Attention
        scale = 1.