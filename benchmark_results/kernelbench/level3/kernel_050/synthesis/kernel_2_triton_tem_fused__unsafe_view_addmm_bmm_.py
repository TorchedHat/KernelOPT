import torch
import triton
import triton.language as tl
import math

# ─── Kernel 1: fused bias-add + split + transpose for Q ───────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=4),
        triton.Config({'BLOCK': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def fused_bias_split_transpose_q_kernel(
    mm_ptr, bias_ptr, out_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    # out shape: [16, 12, 1024, 64]  (B, nh, T, hs)
    # mm shape:  [16384, 2304]  (B*T, 3*C)
    # Q is first 768 elements of dim-1
    # out[b, h, t, d] = mm[b*T+t, h*64+d] + bias[h*64+d]
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    # numel = 16 * 12 * 1024 * 64 = 12582912
    # decode: offs -> (b, h, t, d)
    d = offs % 64
    t = (offs // 64) % 1024
    h = (offs // 65536) % 12
    b = offs // 786432
    # mm index: row = b*1024+t, col = h*64+d (Q slice: col offset 0)
    mm_idx = (b * 1024 + t) * 2304 + h * 64 + d
    bias_idx = h * 64 + d
    val = tl.load(mm_ptr + mm_idx, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val + bias_val, mask=mask)


# ─── Kernel 2: fused bias-add + split + transpose for K (also transpose last 2 dims) ──
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=4),
        triton.Config({'BLOCK': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def fused_bias_split_transpose_k_kernel(
    mm_ptr, bias_ptr, out_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    # out shape: [16, 12, 64, 1024]  (B, nh, hs, T)  -- K transposed
    # mm shape:  [16384, 2304]
    # K is elements [768:1536] of dim-1
    # out[b, h, d, t] = mm[b*T+t, 768 + h*64+d] + bias[768 + h*64+d]
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    # numel = 16 * 12 * 64 * 1024 = 12582912
    # out layout: [B=16, nh=12, hs=64, T=1024]
    t = offs % 1024
    d = (offs // 1024) % 64
    h = (offs // 65536) % 12
    b = offs // 786432
    mm_idx = (b * 1024 + t) * 2304 + 768 + h * 64 + d
    bias_idx = 768 + h * 64 + d
    val = tl.load(mm_ptr + mm_idx, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val + bias_val, mask=mask)


# ─── Kernel 3: fused bias-add + split + transpose for V ───────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=4),
        triton.Config({'BLOCK': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def fused_bias_split_transpose_v_kernel(
    mm_ptr, bias_ptr, out_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    # out shape: [16, 12, 1024, 64]  (B, nh, T, hs)
    # V is elements [1536:2304] of dim-1
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    d = offs % 64
    t = (offs // 64) % 1024
    h = (offs // 65536) % 12
    b = offs // 786432
    mm_idx = (b * 1024 + t) * 2304 + 1536 + h * 64 + d
    bias_idx = 1536 + h * 64 + d
    val = tl.load(mm_ptr + mm_idx, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val + bias_val, mask=mask)


# ─── Kernel 4: Batched GEMM for Q @ K^T ──────────────────────────────────────
# Q: [192, 1024, 64], K^T: [192, 64, 1024] -> att: [192, 1024, 1024]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['B', 'M', 'N', 'K'],
)
@triton.jit
def batched_qk_matmul_kernel(
    Q_ptr, K_ptr, att_ptr, bias_ptr,
    B, M, N, K,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Q: [B, M, K] = [192, 1024, 64]
    # K: [B, K, N] = [192, 64, 1024]  (already transposed)
    # att: [B, M, N] = [192, 1024, 1024]
    # bias: [1, 1, 1024, 1024] -> used as causal mask
    batch = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    Q_base = Q_ptr + batch * M * K
    K_base = K_ptr + batch * K * N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        # Load Q tile [BLOCK_M, BLOCK_K]
        q_ptrs = Q_base + offs_m[:, None] * K + k_offs[None, :]
        q_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # Load K tile [BLOCK_K, BLOCK_N]  (K is already [B, K, N])
        k_ptrs = K_base + k_offs[:, None] * N + offs_n[None, :]
        k_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

        acc = tl.dot(q_tile, k_tile, acc, allow_tf32=False)

    # Scale
    acc = acc * scale

    # Apply causal mask: bias[0,0,m,n] == 0 -> fill with -inf
    # bias shape [1,1,1024,1024], stride [1024*1024, 1024*1024, 1024, 1]
    # We need row indices = offs_m (absolute), col indices = offs_n (absolute)
    b_ptrs = bias_ptr + offs_m[:, None] * 1024 + offs_n[None, :]
    b_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    b_vals = tl.load(b_ptrs, mask=b_mask, other=1.0)  # load causal bias
    # masked_fill: where bias==0, set to -inf
    neg_inf = float('-inf')
    acc = tl.where(b_vals == 0.0, neg_inf, acc)

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store
    out_ptrs = att_ptr + batch * M * N + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ─── Kernel 5: Batched GEMM for att @ V ──────────────────────────────────────
# att: [192, 1024, 1024], V: [192, 1024, 64] -> y: [192, 1024, 64]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['B', 'M', 'N', 'K'],
)
@triton.jit
def batched_av_matmul_kernel(
    att_ptr, V_ptr, out_ptr,
    B, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # att: [B, M, K] = [192, 1024, 1024]
    # V:   [B, K, N] = [192, 1024, 64]
    # out: [B, M, N] = [192, 1024, 64]
    batch = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    att_base = att_ptr + batch * M * K
    V_base = V_ptr + batch * K * N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        # Load att tile [BLOCK_M, BLOCK_K]
        a_ptrs = att_base + offs_m[:, None] * K + k_offs[None, :]
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load V tile [BLOCK_K, BLOCK_N]
        v_ptrs = V_base + k_offs[:, None] * N + offs_n[None, :]
        v_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc = tl.dot(a_tile, v_tile, acc, allow_tf32=False)

    out_ptrs = out_ptr + batch * M * N + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ─── Kernel 6: Linear projection (GEMM) ──────────────────────────────────────
# x: [M, K] = [16384, 768], W: [2304, 768] -> out: [M, 2304]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # x: [M, K], w: [N, K] (weight matrix, we compute x @ w^T)
    # out: [M, N]
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        # Load x tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * K + k_offs[None, :]
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load w tile [BLOCK_K, BLOCK_N]  (w is [N, K], so w^T is [K, N])
        w_ptrs = w_ptr + offs_n[None, :] * K + k_offs[:, None]
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    b_offs = offs_n
    b_mask = offs_n < N
    bias_vals = tl.load(bias_ptr + b_offs, mask=b_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def kernel_function(p_c_attn_weight, p_c_attn_bias, b_bias, x):
    B, T, C = x.shape  # 16, 1024, 768
    n_head = 12
    hs = C // n_head  # 64
    scale = 1.0 / math.sqrt(hs)

    # Step 1: Linear projection x @ W^T + bias -> [B*T, 3*C]
    M_lin = B * T  # 16384
    N_lin = 3 * C  # 2304
    K_lin = C      # 768

    x_2d = x.reshape(M_lin, K_lin).contiguous()
    mm_out = torch.empty((M_lin, N_lin), device=x.device, dtype=torch.float32)

    grid_lin = lambda meta: (triton.cdiv(M_lin, meta['BLOCK_M']) * triton.cdiv(N_lin, meta['BLOCK_N']),)
    linear_kernel[grid_lin](
        x_2d, p_c_attn_weight, p_c_attn_bias, mm_out,
        M_lin, N_lin, K_lin,
    )

    # Step 2: Extract Q, K, V with bias-add + transpose
    # Q: [B, n_head, T, hs] = [16, 12, 1024, 64]
    # K: [B, n_head, hs, T] = [16, 12, 64, 1024]  (transposed)
    # V: [B, n_head, T, hs] = [16, 12, 1024, 64]
    numel_qv = B * n_head * T * hs  # 12582912
    numel_k  = B * n_head * hs * T  # 12582912

    Q = torch.empty((B, n_head, T, hs), device=x.device, dtype=torch.float32)
    K = torch.empty((B, n_head, hs, T), device=x.device, dtype=torch.float32)
    V = torch.empty((B, n_head, T, hs), device=x.device, dtype=torch.float32)

    grid_qv = lambda meta: (triton.cdiv(numel_qv, meta['BLOCK']),)
    fused_bias_split_transpose_q_kernel[grid_qv](mm_out, p_c_attn_bias, Q, numel_qv)
    fused_bias_split_transpose_k_kernel[grid_qv](mm_out, p_c_attn_bias, K, numel_k)
    fused_bias_split_transpose_v_kernel[grid_qv](mm_out, p_c_attn_bias, V, numel_qv)

    # Step 3: att = Q @ K * scale, masked_fill, relu
    # Q: [192, 1024, 64], K: [192, 64, 1024] -> att: [192, 1024, 1024]
    BH = B * n_head  # 192
    att = torch.empty((BH, T, T), device=x.device, dtype=torch.float32)

    Q_3d = Q.reshape(BH, T, hs).contiguous()
    K_3d = K.reshape(BH, hs, T).contiguous()

    # b_bias is [1, 1, 1024, 1024], we need [T, T] slice
    # b_bias[0, 0, :T, :T] - it's 4D
    bias_2d = b_bias[0, 0, :T, :T].contiguous()  # [T, T]

    grid_qk = lambda meta: (
        triton.cdiv(T, meta['BLOCK_M']),
        triton.cdiv(T, meta['BLOCK_N']),
        BH,
    )
    batched_qk_matmul_kernel[grid_qk](
        Q_3d, K_3d, att, bias_2d,
        BH, T, T, hs,
        scale,
    )

    # Step 4: y = att @ V
    # att: [192, 1024, 1024], V: [192, 1024, 64] -> y: [192, 1024, 64]
    V_3d = V.reshape(BH, T, hs).contiguous()
    y_3d = torch.empty((BH, T, hs), device=x.device, dtype=torch.float32)

    grid_av = lambda meta: (
        triton.cdiv(T, meta['BLOCK_M']),
        triton.cdiv(hs, meta['BLOCK_N']),
        BH,
    )
    batched_av_matmul_kernel[grid_av](
        att, V_3d, y_3d,
        BH, T, hs, T,
    )

    # Step 5: reshape to [B, T, C]
    y = y_3d.reshape(B, n_head, T, hs).permute(0, 2, 1, 3).contiguous().reshape(B, T, C)
    return y


def get_inputs():
    B, T, C = 16, 1024, 768
    n_head = 12
    max_seqlen = 1024
    p_c_attn_weight = torch.randn(3 * C, C, device='cuda', dtype=torch.float32)
    p_c_attn_bias = torch.randn(3 * C, device='cuda', dtype=torch.float32)
    b_bias = torch.tril(torch.ones(max_seqlen, max_seqlen, device='cuda')).view(1, 1, max_seqlen, max_seqlen)
    x = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    return [p_c_attn_weight, p_c_attn_bias, b_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    p_c_attn_weight, p_c_attn_bias, b_bias, x = inputs

    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Reference implementation
        import torch.nn as nn
        import torch.nn.functional as F

        B, T, C = x.shape
        n_head = 12
        hs = C // n_head
        scale = 1.0 / math.sqrt(hs)

        # Linear
        qkv = x.reshape(B * T, C) @ p_c_attn_weight.T + p_c_attn_bias
        qkv = qkv.reshape(B, T, 3 * C)
        q_ref, k_ref, v_ref = qkv.split(C, dim=2)

        q_ref = q_ref.view(B, T, n_head, hs).transpose(1, 2)  # [B, nh, T, hs]
        k_ref = k_ref.view(B, T, n_head, hs).transpose(1, 2)  # [B, nh, T, hs]
        v_ref = v_ref.view(B, T, n_head, hs).transpose(1, 2)  # [B, nh, T, hs]

        att_ref = (q_ref @ k_ref.transpose(-2, -1)) * scale
        att_ref = att_ref.masked_fill(b_bias[:, :, :T, :T] == 0, float('-inf'))
        att_ref = F.relu(att_ref)

        y_ref = att_ref @ v_ref
        y_ref = y_ref.transpose(1, 2).contiguous().view(B, T, C)

        max_diff = (output - y_ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, y_ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")