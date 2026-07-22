import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused BN+ReLU epilogue kernel (pointwise)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # channel index
    c = (offsets // HW) % C
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Implicit GEMM Conv2d kernel (NHWC layout internally)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    # dimensions
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    GROUPS,
    M, N, K,
    # strides for X (NCHW)
    s_xn, s_xc, s_xh, s_xw,
    # strides for W (NCHW: out_c, in_c, kh, kw)
    s_wn, s_wc, s_wh, s_ww,
    # strides for Out (NCHW)
    s_on, s_oc, s_oh, s_ow,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    # Group info
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
    
    # Which group does each output channel belong to?
    group_idx = offs_n // GROUP_OUT_C
    local_out_c = offs_n % GROUP_OUT_C
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K = GROUP_IN_C * KERNEL_H * KERNEL_W
    for k_off in range(0, K, BLOCK_K):
        k_range = k_off + tl.arange(0, BLOCK_K)
        
        # Decode k -> (in_c_local, kh, kw)
        kw = k_range % KERNEL_W
        tmp_k = k_range // KERNEL_W
        kh = tmp_k % KERNEL_H
        in_c_local = tmp_k // KERNEL_H
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        # Global in_c (depends on group of output channel)
        # For depthwise (groups=IN_C), each output channel maps to one input channel
        # For grouped conv, group_idx determines offset
        # We need to handle this carefully
        # For simplicity, assume group is determined by output channel
        # group_idx is per output channel (offs_n dimension)
        # But for loading X, we need group per M element too
        # Actually for standard conv: group = 0 for all (GROUPS=1)
        # For depthwise: GROUPS=IN_C, GROUP_IN_C=1, GROUP_OUT_C=1
        # in_c_global = group * GROUP_IN_C + in_c_local
        # But group varies per N (output channel)
        # This makes it complex for a single kernel...
        # Let's handle GROUPS=1 and depthwise separately
        
        in_c_global = in_c_local  # for GROUPS=1
        
        valid_k = k_range < K
        valid_m = (offs_m < M) & (batch_idx < BATCH) & (ih[:, None] >= 0) & (ih[:, None] < IN_H) & (iw[:, None] >= 0) & (iw[:, None] < IN_W)
        
        x_ptrs = X_ptr + batch_idx[:, None] * s_xn + in_c_global[None, :] * s_xc + ih * s_xh + iw * s_xw
        x_mask = valid_m & (valid_k[None, :]) & (in_c_global[None, :] < IN_C)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Weight: shape (OUT_C, IN_C/GROUPS, KH, KW)
        # w_ptrs[k, n] = W[offs_n[n], in_c_local[k], kh[k], kw[k]]
        w_ptrs = W_ptr + offs_n[None, :] * s_wn + in_c_local[:, None] * s_wc + kh[:, None] * s_wh + kw[:, None] * s_ww
        w_mask = (valid_k[:, None]) & (offs_n[None, :] < OUT_C) & (in_c_local[:, None] < GROUP_IN_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Store output
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    out_ptrs = Out_ptr + batch_idx[:, None] * s_on + offs_n[None, :] * s_oc + out_h_idx[:, None] * s_oh + out_w_idx[:, None] * s_ow
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# Depthwise conv kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 9}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 9}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 9}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16, 'BLOCK_K': 9}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 9}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 9}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 9}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 9}, num_warps=8, num_stages=2),
    ],
    key=['M', 'C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH, C, IN_H, IN_W,
    OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # strides for X (NCHW)
    s_xn, s_xc, s_xh, s_xw,
    # strides for W (C, 1, 3, 3)
    s_wc, s_wh, s_ww,
    # strides for Out (NCHW)
    s_on, s_oc, s_oh, s_ow,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,  # kernel elements = 9
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K = 9 (3x3 kernel)
    for k_off in range(0, 9, BLOCK_K):
        k_range = k_off + tl.arange(0, BLOCK_K)
        kw = k_range % 3
        kh = k_range // 3
        
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_k = k_range < 9
        valid_m = (offs_m < M) & (batch_idx < BATCH)
        valid_hw = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        
        # For depthwise: each output channel c uses input channel c
        # x[batch, c, ih, iw] for each c in offs_c
        # We need x[batch_idx[m], offs_c[n], ih[m,k], iw[m,k]]
        # This is a 3D access pattern
        
        # Load x: shape (BLOCK_M, BLOCK_K) for each channel separately
        # Actually we need (BLOCK_M, BLOCK_N, BLOCK_K) which doesn't fit tl.dot
        # Instead, iterate over channels or use a different approach
        
        # For depthwise, let's compute per-channel
        # x_ptrs[m, k] for channel c: X[batch[m], c, ih[m,k], iw[m,k]]
        # This varies per c, so we can't easily vectorize over c with tl.dot
        
        # Alternative: treat as M x N output where for each (m, n) pair:
        # out[m, n] += sum_k X[batch[m], c[n], ih[m,k], iw[m,k]] * W[c[n], 0, kh[k], kw[k]]
        
        # Load W: shape (BLOCK_K, BLOCK_N) = W[offs_c[n], 0, kh[k], kw[k]]
        w_ptrs = W_ptr + offs_c[None, :] * s_wc + kh[:, None] * s_wh + kw[:, None] * s_ww
        w_mask = (valid_k[:, None]) & (offs_c[None, :] < C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # For X, we need to load for each channel separately
        # x_tile[m, k] depends on channel n, so we can't do a single tl.dot
        # We need to loop over channels or use a different strategy
        
        # Strategy: for each channel block, load x and w, accumulate
        # But this would be O(BLOCK_N) loads...
        # 
        # Better: use the fact that for depthwise, we can reshape:
        # Think of it as: for each channel c, compute a 1D conv over spatial
        # 
        # Let's just do it element-wise for now (not optimal but correct)
        for cn in range(BLOCK_N):
            c_idx = pid_n * BLOCK_N + cn
            c_mask = c_idx < C
            
            x_ptrs_c = X_ptr + batch_idx * s_xn + c_idx * s_xc + ih * s_xh + iw * s_xw
            x_mask_c = valid_m[:, None] & valid_hw & valid_k[None, :] & c_mask
            x_tile_c = tl.load(x_ptrs_c, mask=x_mask_c, other=0.0)  # (BLOCK_M, BLOCK_K)
            
            w_col = tl.load(W_ptr + c_idx * s_wc + kh * s_wh + kw * s_ww, 
                           mask=valid_k & c_mask, other=0.0)  # (BLOCK_K,)
            
            # acc[:, cn] += sum_k x_tile_c[:, k] * w_col[k]
            contrib = tl.sum(x_tile_c * w_col[None, :], axis=1)  # (BLOCK_M,)
            # Can't index acc[:, cn] directly in Triton...
            # This approach won't work cleanly
    
    # Store
    out_mask = (offs_m[:, None] < M) & (offs_c[None, :] < C)
    out_ptrs = Out_ptr + batch_idx[:, None] * s_on + offs_c[None, :] * s_oc + out_h_idx[:, None] * s_oh + out_w_idx[:, None] * s_ow
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)