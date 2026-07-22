import torch
import triton
import triton.language as tl
import math

# ─── Kernel 1: QKV projection (GEMM + bias) ───────────────────────────────────
# x: [M, K] = [16*1024, 768], W: [K, N] = [768, 2304], bias: [2304]
# out: [M, N] = [16384, 2304]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def qkv_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
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

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k[None, :] + k * BLOCK_K < K
        m_mask = offs_m[:, None] < M
        x = tl.load(x_ptrs, mask=m_mask & k_mask, other=0.0)
        k_mask2 = offs_k[:, None] + k * BLOCK_K < K
        n_mask = offs_n[None, :] < N
        w = tl.load(w_ptrs, mask=k_mask2 & n_mask, other=0.0)
        acc = tl.dot(x, w, acc, allow_tf32=False)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


# ─── Kernel 2: Attention score: scale + mask + relu ──────────────────────────
# att: [B, nh, T, T], bias: [1, 1, T, T]
# Fused: att * scale, masked_fill where bias==0 with -inf, relu

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_stages=4, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def att_scale_mask_relu_kernel(
    Att_ptr, Bias_ptr, Out_ptr,
    numel,
    T,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # Load attention scores
    att = tl.load(Att_ptr + offs, mask=mask, other=0.0)

    # Compute bias index: bias is [1,1,T,T], att is [B,nh,T,T]
    # bias index = (offs % (T*T))
    bias_idx = offs % (T * T)
    bias_val = tl.load(Bias_ptr + bias_idx, mask=mask, other=0.0)

    # Scale
    att = att * scale

    # Masked fill: where bias == 0, set to -inf
    neg_inf = float('-inf')
    att = tl.where(bias_val == 0.0, neg_inf, att)

    # ReLU
    att = tl.maximum(att, 0.0)

    tl.store(Out_ptr + offs, att, mask=mask)


# ─── Kernel 3: Batched matmul att @ v ─────────────────────────────────────────
# att: [B*nh, T, T], v: [B*nh, T, hs] -> out: [B*nh, T, hs]
# Then reshape to [B, T, C]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['B_nh', 'T', 'hs'],
)
@triton.jit
def batched_matmul_kernel(
    Att_ptr, V_ptr, Out_ptr,
    B_nh, T, hs,
    stride_att_b, stride_att_m, stride_att_k,
    stride_v_b,   stride_v_k,   stride_v_n,
    stride_out_b, stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # pid layout: (batch_head, tile_m, tile_n) flattened
    pid_b = tl.program_id(2)
    pid   = tl.program_id(0)
    pid_n = tl.program_id(1)

    grid_m = tl.cdiv(T, BLOCK_M)

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    att_ptrs = Att_ptr + pid_b * stride_att_b + offs_m[:, None] * stride_att_m + offs_k[None, :] * stride_att_k
    v_ptrs   = V_ptr   + pid_b * stride_v_b   + offs_k[:, None] * stride_v_k   + offs_n[None, :] * stride_v_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(T, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        m_mask = offs_m[:, None] < T
        k_mask_a = k_offs[None, :] < T
        k_mask_v = k_offs[:, None] < T
        n_mask = offs_n[None, :] < hs

        a = tl.load(att_ptrs, mask=m_mask & k_mask_a, other=0.0)
        v = tl.load(v_ptrs,   mask=k_mask_v & n_mask, other=0.0)
        acc = tl.dot(a, v, acc, allow_tf32=False)
        att_ptrs += BLOCK_K * stride_att_k
        v_ptrs   += BLOCK_K * stride_v_k

    out_ptrs = Out_ptr + pid_b * stride_out_b + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask = (offs_m[:, None] < T) & (offs_n[None, :] < hs)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


# ─── QK matmul: q @ k^T ───────────────────────────────────────────────────────
# q: [B*nh, T, hs], k^T: [B*nh, hs, T] -> att: [B*nh, T, T]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['B_nh', 'T', 'hs'],
)
@triton.jit
def qk_matmul_kernel(
    Q_ptr, K_ptr, Out_ptr,
    B_nh, T, hs,
    stride_q_b, stride_q_m, stride_q_k,
    stride_k_b, stride_k_k, stride_k_n,
    stride_out_b, stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q_ptrs = Q_ptr + pid_b * stride_q_b + offs_m[:, None] * stride_q_m + offs_k[None, :] * stride_q_k
    k_ptrs = K_ptr + pid_b * stride_k_b + offs_k[:, None] * stride_k_k + offs_n[None, :] * stride_k_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(hs, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        m_mask = offs_m[:, None] < T
        k_mask_q = k_offs[None, :] < hs
        k_mask_k = k_offs[:, None] < hs
        n_mask = offs_n[None, :] < T

        q = tl.load(q_ptrs, mask=m_mask & k_mask_q, other=0.0)
        k_tile = tl.load(k_ptrs, mask=k_mask_k & n_mask, other=0.0)
        acc = tl.dot(q, k_tile, acc, allow_tf32=False)
        q_ptrs += BLOCK_K * stride_q_k
        k_ptrs += BLOCK_K * stride_k_k

    out_ptrs = Out_ptr + pid_b * stride_out_b + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask = (offs_m[:, None] < T) & (offs_n[None, :] < T)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


def kernel_function(p_c_attn_weight, p_c_attn_bias, b_bias, x):
    B, T, C = x.shape
    n_head = 12
    hs = C // n_head  # 64
    N_qkv = 3 * C     # 2304

    # Step 1: QKV projection [B*T, C] @ [C, 3C] + bias -> [B*T, 3C]
    x_2d = x.reshape(B * T, C).contiguous()
    # W is [2304, 768], we need [768, 2304]
    W = p_c_attn_weight  # [2304, 768] -> treat as W^T, so x @ W^T
    # Actually linear is x @ W.T + b where W is [out, in]
    # So we compute x_2d @ W.T where W.T is [768, 2304]
    # In our kernel: X[M,K] @ W[K,N] where K=768, N=2304
    # W_ptr points to p_c_attn_weight which is [2304, 768]
    # stride_wk=1 (inner dim of W^T = row of W), stride_wn=768 (col of W^T = row stride of W)
    M_qkv = B * T
    K_qkv = C
    N_qkv_out = N_qkv

    qkv_out = torch.empty((M_qkv, N_qkv_out), device=x.device, dtype=torch.float32)

    grid_qkv = lambda meta: (
        triton.cdiv(M_qkv, meta['BLOCK_M']) * triton.cdiv(N_qkv_out, meta['BLOCK_N']),
    )
    qkv_gemm_kernel[grid_qkv](
        x_2d, p_c_attn_weight, p_c_attn_bias, qkv_out,
        M_qkv, N_qkv_out, K_qkv,
        x_2d.stride(0), x_2d.stride(1),
        # W is [2304, 768]: stride(0)=768, stride(1)=1
        # We want X @ W^T: W^T[k,n] = W[n,k]
        # So stride_wk = W.stride(1) = 1 (k indexes columns of W = rows of W^T)
        # stride_wn = W.stride(0) = 768 (n indexes rows of W = cols of W^T)
        p_c_attn_weight.stride(1), p_c_attn_weight.stride(0),
    )

    # qkv_out: [B*T, 3C] -> [B, T, 3C]
    qkv_out_3d = qkv_out.reshape(B, T, N_qkv_out)

    # Split into Q, K, V: each [B, T, C]
    # Q = qkv_out_3d[:, :, :C]
    # K = qkv_out_3d[:, :, C:2C]
    # V = qkv_out_3d[:, :, 2C:]
    # Reshape to [B, nh, T, hs] and make contiguous
    # q: [B, T, C] -> [B, T, nh, hs] -> [B, nh, T, hs] -> [B*nh, T, hs]
    q = qkv_out_3d[:, :, :C].reshape(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous().reshape(B * n_head, T, hs)
    k = qkv_out_3d[:, :, C:2*C].reshape(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous().reshape(B * n_head, T, hs)
    v = qkv_out_3d[:, :, 2*C:].reshape(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous().reshape(B * n_head, T, hs)

    # Step 2: att = q @ k^T  [B*nh, T, T]
    B_nh = B * n_head
    att = torch.empty((B_nh, T, T), device=x.device, dtype=torch.float32)

    # k^T: [B*nh, hs, T] - k is [B*nh, T, hs], k^T has stride_k=T, stride_n=1
    # k[b, t, h] -> k^T[b, h, t]: stride_k_b = T*hs, stride_k_k = hs (h dim), stride_k_n = 1 (t dim)
    # Wait: k is [B*nh, T, hs] with strides [T*hs, hs, 1]
    # k^T[b, h, t] = k[b, t, h]
    # So to access k^T: ptr = k_ptr + b*T*hs + h*1 + t*hs
    # stride_k_b = T*hs, stride_k_k = 1 (h index), stride_k_n = hs (t index)

    grid_qk = lambda meta: (
        triton.cdiv(T, meta['BLOCK_M']),
        triton.cdiv(T, meta['BLOCK_N']),
        B_nh,
    )
    qk_matmul_kernel[grid_qk](
        q, k, att,
        B_nh, T, hs,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(2), k.stride(1),  # k^T: stride_k_b, stride_k_k(hs dim), stride_k_n(T dim)
        att.stride(0), att.stride(1), att.stride(2),
    )

    # Step 3: scale + mask + relu
    scale = 1.0 / math.sqrt(hs)
    att_out = torch.empty_like(att)
    numel = B_nh * T * T

    # b_bias is [1, 1, 1024, 1024], we need [:,:,:T,:T]
    # Since T=1024 here, b_bias is exactly [1,1,T,T]
    # Flatten to [T*T] for indexing
    bias_flat = b_bias.reshape(T * T)  # [T*T]

    grid_mask = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    att_scale_mask_relu_kernel[grid_mask](
        att, bias_flat, att_out,
        numel, T, scale,
    )

    # Step 4: y = att_out @ v  [B*nh, T, T] @ [B*nh, T, hs] -> [B*nh, T, hs]
    y = torch.empty((B_nh, T, hs), device=x.device, dtype=torch.float32)

    grid_av = lambda meta: (
        triton.cdiv(T, meta['BLOCK_M']),
        triton.cdiv(hs, meta['BLOCK_N']),
        B_nh,
    )
    batched_matmul_kernel[grid_av](
        att_out, v, y,
        B_nh, T, hs,
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
    )

    # Step 5: reshape [B*nh, T, hs] -> [B, nh, T, hs] -> [B, T, nh, hs] -> [B, T, C]
    y_out = y.reshape(B, n_head, T, hs).permute(0, 2, 1, 3).contiguous().reshape(B, T, C)
    return y_out


def get_inputs():
    B = 16
    T = 1024
    C = 768
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
        B, T, C = x.shape
        n_head = 12
        hs = C // n_head

        # QKV projection
        qkv = x.reshape(B * T, C) @ p_c_attn_weight.T + p_c_attn_bias
        qkv = qkv.reshape(B, T, 3 * C)
        q_ref = qkv[:, :, :C].reshape(B, T, n_head, hs).permute(0, 2, 1, 3)
        k_ref = qkv[:, :, C:2*C].reshape(B, T, n_head, hs).permute(0, 2, 1, 3)
        v_ref = qkv[:, :, 2*C:].reshape(B, T, n_head, hs).permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(hs)
        att_ref = (q_ref @ k_ref.transpose(-2, -1)) * scale
        att_ref = att_ref.masked_fill(b_bias[:, :, :T, :T] == 0, float('-inf'))
        att_ref = torch.relu(att_ref)
        y_ref = att_ref @ v_ref
        y_ref = y_ref.transpose(1, 2).contiguous().reshape(B, T, C)

        assert torch.allclose(output, y_ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - y_ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")