import torch
import triton
import triton.language as tl
import math

# ─── Kernel 1: Permute + GEMM for in_proj (fused as separate ops) ───────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_permute_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    # X is (B, C, HW) stored as (B, C, HW), we access as (HW, B, C) logically
    # X shape: B=2, C=128, HW=16384
    # We want Out[i, b, n] = sum_k X[b, k, i] * W[n, k] + B[n]
    # Out shape: (HW=16384, B=2, N=384)
    HW, B, C, N,
    M,  # M = HW * B = 32768
    K,  # K = C = 128
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # m index: m = hw_idx * B + b_idx
    # hw_idx = offs_m // B, b_idx = offs_m % B
    hw_idx = offs_m // B  # (BLOCK_M,)
    b_idx = offs_m % B    # (BLOCK_M,)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K
        
        # Load X tile: X[b_idx, k_offs, hw_idx] with strides
        # X ptr: b_idx * stride_xb + k_offs * stride_xc + hw_idx * stride_xhw
        x_ptrs = X_ptr + (b_idx[:, None] * stride_xb + 
                          k_offs[None, :] * stride_xc + 
                          hw_idx[:, None] * stride_xhw)
        m_mask = offs_m < M
        x_tile = tl.load(x_ptrs, 
                         mask=m_mask[:, None] & k_mask[None, :], 
                         other=0.0)
        
        # Load W tile: W[offs_n, k_offs] -> shape (BLOCK_N, BLOCK_K), need (BLOCK_K, BLOCK_N)
        w_ptrs = W_ptr + (offs_n[None, :] * K + k_offs[:, None])
        n_mask = offs_n < N
        w_tile = tl.load(w_ptrs,
                         mask=k_mask[:, None] & n_mask[None, :],
                         other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    # Store: Out[hw_idx, b_idx, offs_n] with shape (HW, B, N), stride (B*N, N, 1)
    out_ptrs = Out_ptr + (hw_idx[:, None] * (B * N) + b_idx[:, None] * N + offs_n[None, :])
    m_mask = offs_m < M
    n_mask = offs_n < N
    tl.store(out_ptrs, acc.to(tl.float32), mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, Bias_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        m_mask = offs_m < M
        n_mask = offs_n < N
        
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_tile = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        acc = tl.dot(a_tile, b_tile, acc)
    
    if HAS_BIAS:
        bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
    
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    m_mask = offs_m < M
    n_mask = offs_n < N
    tl.store(out_ptrs, acc.to(tl.float32), mask=m_mask[:, None] & n_mask[None, :])


# ─── Kernel 2: Softmax over last dim ────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 16384}, num_warps=16, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(
    Out_ptr, In_ptr,
    M, N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    
    row_ptr = In_ptr + row * stride_m
    out_row_ptr = Out_ptr + row * stride_m
    
    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        vals = tl.load(row_ptr + cols, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))
    
    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        vals = tl.load(row_ptr + cols, mask=mask, other=float('-inf'))
        row_sum += tl.sum(tl.exp(vals - row_max), axis=0)
    
    # Normalize and store
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        vals = tl.load(row_ptr + cols, mask=mask, other=float('-inf'))
        out = tl.exp(vals - row_max) / row_sum
        tl.store(out_row_ptr + cols, out, mask=mask)


# ─── Kernel 3: LayerNorm ─────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def layernorm_kernel(
    Out_ptr, In_ptr, Residual_ptr, Weight_ptr, Bias_ptr,
    M, N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    
    in_ptr = In_ptr + row * N
    res_ptr = Residual_ptr + row * N
    out_ptr = Out_ptr + row * N
    
    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(in_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x + r, axis=0)
    mean = mean / N
    
    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(in_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = (x + r) - mean
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(in_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(Weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(Bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = ((x + r) - mean) * rstd * w + b
        tl.store(out_ptr + cols, out.to(tl.float32), mask=mask)


# ─── BMM kernel ──────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K', 'num_batches'],
)
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_ob, stride_om, stride_on,
    num_batches,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_b >= num_batches:
        return
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    A_batch = A_ptr + pid_b * stride_ab
    B_batch = B_ptr + pid_b * stride_bb
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        m_mask = offs_m < M
        n_mask = offs_n < N
        
        a_ptrs = A_batch + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        b_ptrs = B_batch + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_tile = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        acc = tl.dot(a_tile, b_tile, acc)
    
    out_ptrs = Out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    m_mask = offs_m < M
    n_mask = offs_n < N
    tl.store(out_ptrs, acc.to(tl.float32), mask=m_mask[:, None] & n_mask[None, :])


def cdiv(a, b):
    return (a + b - 1) // b


def kernel_function(
    p_attn_in_proj_bias,
    p_attn_in_proj_weight,
    p_attn_out_proj_weight,
    p_attn_out_proj_bias,
    p_norm_weight,
    p_norm_bias,
    x,
):
    B, C, H, W = x.shape
    # B=2, C=128, H=128, W=128
    HW = H * W  # 16384
    embed_dim = C  # 128
    num_heads = 4  # wait, the computation graph shows 8 heads with head_dim=32
    # Actually from the graph: view to [16384, 8, 32] means 8 heads, head_dim=32
    # But num_heads=4 in the model... Let me re-check
    # The graph shows: view.default -> shape=[16384, 8, 32]
    # So num_heads=8? But model says num_heads=4...
    # Wait, looking at the graph more carefully:
    # select.int -> shape=[16384, 2, 128] (Q, K, V each)
    # view.default -> shape=[16384, 8, 32]  <- 8 heads, head_dim=32
    # But embed_dim=128, 128/8=16... that doesn't match
    # Actually 128/4=32, so 4 heads with head_dim=32
    # But the graph says shape=[16384, 8, 32]... 
    # Wait: batch=2, so 2*4=8 heads? No...
    # Let me re-read: the bmm is [8, 16384, 16384]
    # Q shape after view: [16384, 8, 32], after transpose: [8, 16384, 32]
    # So 8 heads total. But embed_dim=128, 128/8=16 not 32...
    # Unless num_heads=4 in the model but the batch dimension is folded in?
    # Actually: B=2 batches, 4 heads each -> 8 "batch*heads" for bmm
    # That makes sense! The bmm treats batch*heads as the batch dim
    # head_dim = embed_dim // num_heads = 128 // 4 = 32
    # num_heads_total = B * num_heads = 2 * 4 = 8
    
    num_heads_model = 4
    head_dim = embed_dim // num_heads_model  # 32
    num_heads_total = B * num_heads_model  # 8
    scale = head_dim ** -0.5  # 1/sqrt(32) = 0.1767...
    
    device = x.device
    dtype = x.dtype
    
    # Step 1: x.view(B, C, H*W).permute(2, 0, 1) -> (HW, B, C)
    # Then linear: (HW*B, C) @ W.T + bias -> (HW*B, 3*C)
    # W shape: (384, 128), bias shape: (384,)
    
    # x is (B, C, HW) with strides (C*HW, HW, 1)
    x_flat = x.reshape(B, C, HW)  # (B, C, HW)
    
    # We need (HW*B, C) where row i*B+b corresponds to x[b, :, i]
    # Use gemm_permute_kernel to do this in one step
    
    # Actually, let's just do it simply:
    # x_perm: (HW, B, C) contiguous
    x_perm = x_flat.permute(2, 0, 1).contiguous()  # (HW, B, C)
    x_2d = x_perm.reshape(HW * B, C)  # (32768, 128)
    
    # in_proj: (32768, 128) @ (128, 384) + (384,) -> (32768, 384)
    in_proj_out = torch.empty(HW * B, 3 * embed_dim, device=device, dtype=dtype)
    M_ip = HW * B
    N_ip = 3 * embed_dim
    K_ip = embed_dim
    grid_ip = (cdiv(M_ip, 64), cdiv(N_ip, 128))
    gemm_kernel[grid_ip](
        x_2d, p_attn_in_proj_weight, p_attn_in_proj_bias, in_proj_out,
        M_ip, N_ip, K_ip,
        x_2d.stride(0), x_2d.stride(1),
        p_attn_in_proj_weight.stride(0), p_attn_in_proj_weight.stride(1),
        in_proj_out.stride(0), in_proj_out.stride(1),
        True,
    )
    # in_proj_out: (32768, 384) = (HW*B, 3*C)
    # Reshape to (HW, B, 3, C) then split Q, K, V
    in_proj_out_4d = in_proj_out.reshape(HW, B, 3, embed_dim)
    
    # Q = in_proj_out_4d[:, :, 0, :] -> (HW, B, C)
    # K = in_proj_out_4d[:, :, 1, :] -> (HW, B, C)
    # V = in_proj_out_4d[:, :, 2, :] -> (HW, B, C)
    Q = in_proj_out_4d[:, :, 0, :].contiguous()  # (HW, B, C)
    K = in_proj_out_4d[:, :, 1, :].contiguous()  # (HW, B, C)
    V = in_proj_out_4d[:, :, 2, :].contiguous()  # (HW, B, C)
    
    # Reshape for multi-head: (HW, B, num_heads, head_dim)
    # Then permute to (B*num_heads, HW, head_dim) for bmm
    Q = Q.reshape(HW, B * num_heads_model, head_dim).permute(1, 0, 2).contiguous()  # (8, HW, 32)
    K = K.reshape(HW, B * num_heads_model, head_dim).permute(1, 0, 2).contiguous()  # (8, HW, 32)
    V = V.reshape(HW, B * num_heads_model, head_dim).permute(1, 0, 2).contiguous()  # (8, HW, 32)
    
    # Scale Q
    Q = Q * scale
    
    # Attention scores: Q @ K.T -> (8, HW, HW)
    # K needs to be transposed: (8, 32, HW)
    K_t = K.transpose(1, 2).contiguous()  # (8, 32, HW)
    
    attn_scores = torch.empty(num_heads_total, HW, HW, device=device, dtype=dtype)
    grid_bmm1 = (cdiv(HW, 32), cdiv(HW, 32), num_heads_total)
    bmm_kernel[grid_bmm1](
        Q, K_t, attn_scores,
        HW, HW, head_dim,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K_t.stride(0), K_t.stride(1), K_t.stride(2),
        attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2),
        num_heads_total,
    )
    
    # Softmax over last dim
    attn_weights = torch.empty_like(attn_scores)
    M_sm = num_heads_total * HW
    grid_sm = (M_sm,)
    softmax_kernel[grid_sm](
        attn_weights, attn_scores,
        M_sm, HW,
        HW,
    )
    
    # Attention output: attn_weights @ V -> (8, HW, 32)
    attn_out = torch.empty(num_heads_total, HW, head_dim, device=device, dtype=dtype)
    grid_bmm2 = (cdiv(HW, 32), cdiv(head_dim, 32), num_heads_total)
    bmm_kernel[grid_bmm2](
        attn_weights, V, attn_out,
        HW, head_dim, HW,
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
        num_heads_total,
    )
    
    # Reshape: (8, HW, 32) -> (HW, 8, 32) -> (HW*B, C) = (32768, 128)
    attn_out = attn_out.permute(1, 0, 2).contiguous()  # (HW, 8, 32)
    attn_out_2d = attn_out.reshape(HW * B, embed_dim)  # (32768, 128)
    
    # out_proj: (32768, 128) @ (128, 128).T + (128,) -> (32768, 128)
    out_proj_out = torch.empty(HW * B, embed_dim, device=device, dtype=dtype)
    grid_op = (cdiv(HW * B, 64), cdiv(embed_dim, 128))
    gemm_kernel[grid_op](
        attn_out_2d, p_attn_out_proj_weight, p_attn_out_proj_bias, out_proj_out,
        HW * B, embed_dim, embed_dim,
        attn_out_2d.stride(0), attn_out_2d.stride(1),
        p_attn_out_proj_weight.stride(0), p_attn_out_proj_weight.stride(1),
        out_proj_out.stride(0), out_proj_out.stride(1),
        True,
    )
    
    # Reshape out_proj_out to (HW, B, C)
    out_proj_4d = out_proj_out.reshape(HW, B, embed_dim)
    
    # Now we need the mean over heads dimension
    # From the graph: view.default -> shape=[2, 4, 16384, 16384] then mean.dim -> shape=[2, 16384, 16384]
    # Wait, that seems odd. Let me re-read...
    # Actually looking at the graph again:
    # bmm.default -> shape=[8, 16384, 16384]  (attn_scores)
    # softmax.int -> shape=[8, 16384, 16384]  (attn_weights)
    # bmm.default -> shape=[8, 16384, 32]     (attn_out)
    # ...
    # view.default -> shape=[2, 4, 16384, 16384]  <- this is attn_weights reshaped
    # mean.dim -> shape=[2, 16384, 16384]  <- mean over heads dim
    # But this is just the attention weights for visualization, not used in the output!
    # The actual output path is:
    # out_proj_out -> reshape -> add residual -> layernorm -> permute -> view
    
    # The mean.dim output is the second return value (attn_weights averaged)
    # but in the model, attn_output, _ = self.attn(x, x, x) - the _ is discarded
    # So we don't need to compute it for correctness
    
    # Residual + LayerNorm
    # x_perm is (HW, B, C) = (16384, 2, 128)
    # out_proj_4d is (HW, B, C) = (16384, 2, 128)
    # norm(out_proj_4d + x_perm)
    
    ln_out = torch.empty(HW, B, embed_dim, device=device, dtype=dtype)
    M_ln = HW * B
    grid_ln = (M_ln,)
    
    # Flatten for layernorm: treat each (HW*B) row as a separate sample
    out_proj_flat = out_proj_4d.reshape(HW * B, embed_dim)