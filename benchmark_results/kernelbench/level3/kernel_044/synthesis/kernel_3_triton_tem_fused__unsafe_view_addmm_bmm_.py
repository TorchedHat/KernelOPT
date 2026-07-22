import torch
import triton
import triton.language as tl
import math

# ─── Fused LayerNorm kernel ───────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_N': 512},  num_warps=4),
        triton.Config({'BLOCK_N': 256},  num_warps=4),
        triton.Config({'BLOCK_N': 128},  num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, eps,
    stride_xm,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * stride_xm
    out_ptr = out_ptr + row * stride_xm

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = tl.math.rsqrt(var + eps)

    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, y.to(tl.float32), mask=mask)


# ─── Fused QKV projection + reshape kernel ───────────────────────────────────
# We'll do the matmul with tl.dot and add bias
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k + k * BLOCK_K < K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask)


# ─── Fused attention score kernel: QK^T * scale + causal mask + softmax ──────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64,  'BLOCK_S': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 128, 'BLOCK_S': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 64,  'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 128, 'BLOCK_S': 128}, num_warps=8, num_stages=2),
    ],
    key=['T', 'HS'],
)
@triton.jit
def attn_softmax_kernel(
    Q_ptr, K_ptr, V_ptr, bias_ptr, Out_ptr,
    B, NH, T, HS,
    scale,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Each program handles one (batch, head, query_tile) combination
    pid = tl.program_id(0)
    num_tiles_t = tl.cdiv(T, BLOCK_T)
    bh = pid // num_tiles_t
    tile_t = pid % num_tiles_t

    b = bh // NH
    h = bh % NH

    # Query tile indices
    q_start = tile_t * BLOCK_T
    q_offs = q_start + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)

    # Load Q tile [BLOCK_T, BLOCK_D]
    Q_base = Q_ptr + b * stride_qb + h * stride_qh
    q_mask = q_offs[:, None] < T
    d_mask = d_offs[None, :] < HS
    Q = tl.load(Q_base + q_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                mask=q_mask & d_mask, other=0.0)

    # Online softmax state
    m_i = tl.full([BLOCK_T], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_T], dtype=tl.float32)
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    K_base = K_ptr + b * stride_kb + h * stride_kh
    V_base = V_ptr + b * stride_vb + h * stride_vh

    # Iterate over key/value tiles
    for s_start in range(0, T, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < T

        # Load K tile [BLOCK_S, BLOCK_D]
        K_tile = tl.load(K_base + s_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd,
                         mask=s_mask[:, None] & (d_offs[None, :] < HS), other=0.0)

        # Compute QK^T [BLOCK_T, BLOCK_S]
        scores = tl.dot(Q, tl.trans(K_tile)) * scale

        # Apply causal mask: q_offs >= s_offs means valid
        causal_mask = q_offs[:, None] >= s_offs[None, :]
        # Also apply the bias (which is 0 for valid, -inf for masked)
        # bias shape: [1, 1, 1024, 1024], we need [q_offs, s_offs]
        bias_vals = tl.load(bias_ptr + q_offs[:, None] * 1024 + s_offs[None, :],
                            mask=(q_offs[:, None] < 1024) & (s_offs[None, :] < 1024), other=0.0)
        # bias is 1 for valid, 0 for masked (lower triangular)
        scores = tl.where(bias_vals == 0.0, float('-inf'), scores)
        scores = tl.where(s_mask[None, :], scores, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp(m_i - m_new)
        p = tl.math.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V tile [BLOCK_S, BLOCK_D]
        V_tile = tl.load(V_base + s_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd,
                         mask=s_mask[:, None] & (d_offs[None, :] < HS), other=0.0)
        acc = tl.dot(p.to(V_tile.dtype), V_tile, acc=acc)
        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output [BLOCK_T, BLOCK_D]
    Out_base = Out_ptr + b * stride_ob + h * stride_oh
    out_mask = (q_offs[:, None] < T) & (d_offs[None, :] < HS)
    tl.store(Out_base + q_offs[:, None] * stride_ot + d_offs[None, :] * stride_od,
             acc.to(tl.float32), mask=out_mask)


# ─── Fused GELU kernel ────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(out_ptr + offs, y.to(tl.float32), mask=mask)


# ─── Fused residual add kernel ────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def add_kernel(
    a_ptr, b_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


def run_layer_norm(x, weight, bias, eps=1e-5):
    M, N = x.shape[0] * x.shape[1], x.shape[2]
    out = torch.empty_like(x)
    x_2d = x.view(M, N)
    out_2d = out.view(M, N)
    grid = (M,)
    layer_norm_kernel[grid](
        x_2d, weight, bias, out_2d,
        M, N, eps,
        x_2d.stride(0),
    )
    return out


def run_matmul_bias(A, B, bias):
    # A: [M, K], B: [N, K] (weight matrix), bias: [N]
    # Output: [M, N]
    M, K = A.shape
    N = B.shape[0]
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    # B is stored as [N, K], we want A @ B.T
    # So stride_bk=1, stride_bn=K (treating B as [N,K] and accessing B[n,k])
    grid = lambda meta: (tl.cdiv(M, meta['BLOCK_M']) * tl.cdiv(N, meta['BLOCK_N']),)
    matmul_bias_kernel[grid](
        A, B, bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(1), B.stride(0),  # B is [N,K], we want B^T so stride_bk=B.stride(1), stride_bn=B.stride(0)
        C.stride(0), C.stride(1),
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

    # ── 1. LayerNorm 1 ────────────────────────────────────────────────────────
    ln1_out = run_layer_norm(x, p_ln_1_weight, p_ln_1_bias)

    # ── 2. QKV projection: [B*T, C] @ [C, 3C]^T + bias ──────────────────────
    ln1_2d = ln1_out.view(B * T, C)
    qkv = run_matmul_bias(ln1_2d, p_attn_c_attn_weight, p_attn_c_attn_bias)
    # qkv: [B*T, 2304]
    qkv_3d = qkv.view(B, T, 3 * C)

    # Split Q, K, V
    q = qkv_3d[:, :, :C].view(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()
    k = qkv_3d[:, :, C:2*C].view(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()
    v = qkv_3d[:, :, 2*C:].view(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()
    # q, k, v: [B, NH, T, HS]

    # ── 3. Flash Attention ────────────────────────────────────────────────────
    attn_out = torch.empty(B, n_head, T, hs, device=x.device, dtype=x.dtype)
    scale = 1.0 / math.sqrt(hs)

    # bias_ptr points to the causal mask [1, 1, 1024, 1024]
    # We need to access bias[0, 0, q_idx, k_idx]
    bias_2d = b_attn_bias.view(1024, 1024)  # [1024, 1024]

    # BLOCK_D must be power of 2 >= hs=96, so use 128
    BLOCK_D = 128

    num_tiles_t = math.ceil(T / 64)  # using BLOCK_T=64 as default
    grid_attn = (B * n_head * num_tiles_t,)

    # We need to pick BLOCK_T and BLOCK_S that are powers of 2
    # Use BLOCK_T=64, BLOCK_S=64, BLOCK_D=128
    # But autotune will handle this

    # Actually let's just call with fixed config for now
    # The autotune kernel needs BLOCK_D as constexpr too
    # Let's restructure: use a simpler approach

    # Use flash attention with BLOCK_T=64, BLOCK_S=64
    BT = 64
    BS = 64
    num_tiles = math.ceil(T / BT)
    grid_attn = (B * n_head * num_tiles,)

    attn_softmax_kernel[grid_attn](
        q, k, v, bias_2d, attn_out,
        B, n_head, T, hs,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
        BLOCK_T=BT, BLOCK_S=BS, BLOCK_D=BLOCK_D,
    )

    # ── 4. Reshape attention output ───────────────────────────────────────────
    y = attn_out.permute(0, 2, 1, 3).contiguous().view(B * T, C)

    # ── 5. Output projection ──────────────────────────────────────────────────
    proj_out = run_matmul_bias(y, p_attn_c_proj_weight, p_attn_c_proj_bias)
    proj_out = proj_out.view(B, T, C)

    # ── 6. Residual add ───────────────────────────────────────────────────────
    x = x + proj_out  # [B, T, C]

    # ── 7. LayerNorm 2 ────────────────────────────────────────────────────────
    ln2_out = run_layer_norm(x, p_ln_2_weight, p_ln_2_bias)

    # ── 8. MLP FC: [B*T, C] @ [3072, C]^T + bias ─────────────────────────────
    ln2_2d = ln2_out.view(B * T, C)
    fc_out = run_matmul_bias(ln2_2d, p_mlp_c_fc_weight, p_mlp_c_fc_bias)
    # fc_out: [B*T, 3072]

    # ── 9. GELU activation ────────────────────────────────────────────────────
    gelu_out = torch.empty_like(fc_out)
    N_gelu = fc_out.numel()
    grid_gelu = lambda meta: (tl.cdiv(N_gelu, meta['BLOCK_SIZE']),)
    gelu_kernel[grid_gelu](fc_out, gelu_out, N_gelu)

    # ── 10. MLP proj: [B*T, 3072] @ [768, 3072]^T + bias ────────────────────
    proj2_out = run_matmul_bias(gelu_out, p_mlp_c_proj_weight, p_mlp_c_proj_bias)
    proj2_out = proj2_out.view(B, T, C)

    # ── 11. Final residual add ────────────────────────────────────────────────
    out = x + proj2_out

    return out


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'
    B, T, C = 128, 512, 768
    n_head = 8
    max_seqlen = 1024

    p_ln_1_weight = torch.randn(C, device=device)
    p_ln_1_bias = torch.randn(C, device=device)
    p_attn_c_attn_weight = torch.randn(3*C, C, device=device) * 0.02
    p_attn_c_attn_bias = torch.randn(3*C, device=device) * 0.02
    p_attn_c_proj_weight = torch.randn(C, C, device=device) * 0.02
    p_attn_c_proj_bias = torch.randn(C, device=device) * 0.02
    p_ln_2_weight = torch.randn(C, device=device)
    p_ln_2_bias = torch.randn(C, device=device)
    p_mlp_c_fc_weight = torch.randn(3072, C, device=device) * 0.02
    p_mlp_c_fc_bias = torch.randn(3072, device=device) * 0.02
    p_mlp_c_proj_weight = torch.randn(C, 3072, device=device) * 0.02
    p_mlp_c_proj_bias = torch.randn(C, device=device) * 0.02
    b_attn_bias = torch.tril(torch.ones(max_seqlen, max_seqlen, device=device)).view(1, 1, max_seqlen, max_seqlen)
    x = torch.randn(B, T, C, device=device) * 0.02

    return [
        p_ln_1_weight, p_ln_1_bias,
        p_attn_c_attn_weight, p_attn_c_attn_bias,
        p_attn_c_proj_weight, p_attn_c_proj_bias,
        p_ln_2_weight, p_ln_2_bias,
        p_mlp_c_fc_weight, p_mlp_c_fc_bias,
        p_mlp_c_proj_weight, p_mlp_c_proj_bias,
        b_attn_bias,
        x,
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
        import torch.nn.functional as F

        (p_ln_1_weight, p_ln_1_bias,
         p_attn_c_attn_weight, p_attn_c_attn_bias,
         p_attn_c_proj_weight, p_attn_c_proj_bias,
         p_ln_2_weight, p_ln_2_bias,
         p_mlp_c_fc_weight, p_mlp_c_fc_bias,
         p_mlp_c_proj_weight, p_mlp_c_proj_bias,
         b_attn_bias,
         x) = inputs

        B, T, C = x.shape
        n_head = 8
        hs = C // n_head

        # LayerNorm 1
        ln1 = F.layer_norm(x, [C], p_ln_1_weight, p_ln_1_bias)

        # QKV
        qkv = F.linear(ln1, p_attn_c_attn_weight, p_attn_c_attn_bias)
        q_r, k_r, v_r = qkv.split(C, dim=2)
        q_r = q_r.view(B, T, n_head, hs).transpose(1, 2)
        k_r = k_r.view(B, T, n_head, hs).transpose(1, 2)
        v_r = v_r.view(B, T, n_head, hs).transpose(1, 2)

        # Attention
        att = (q_r @ k_r.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(b_attn_bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y_r = att @ v_r
        y_r = y_r.transpose(1, 2).contiguous().view(B, T, C)

        # Output proj
        proj_r = F.linear(y_r, p_attn_c_proj_weight, p_attn_c_proj_bias)
        x_r = x + proj_r

        # LayerNorm 2
        ln2 = F.layer_norm(x_r, [C], p_ln_2_weight, p_ln_2_bias)

        # MLP
        fc = F.linear(ln2, p_mlp_c_fc_weight, p_mlp_c_fc_bias)
        # GELU
        gelu_r = 0.5 * fc * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (fc + 0.044715 * fc.pow(3.0))))
        proj2_r = F.linear(gelu_r, p_mlp_c_proj_weight, p_mlp_c_proj_bias)
        ref = x_r + proj2_r

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-2), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")