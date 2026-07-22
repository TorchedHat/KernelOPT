import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernel: fused conv2d (1x1) + bias + relu
# For 1x1 convolutions: GEMM style
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc, allow_tf32=False)
    
    # Add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    # ReLU
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)
    
    # Store
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.float32), mask=y_mask)


def conv1x1_bias_relu(x, weight, bias, apply_relu=True):
    """
    """
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    M = N * H * W
    K = C_in
    
    # Reshape x to (M, K) - need contiguous in channel-last or standard layout
    # x is (N, C_in, H, W), we want (N*H*W, C_in)
    x_2d = x.permute(0, 2, 3, 1).contiguous().reshape(M, K)
    w_2d = weight.reshape(C_out, K)  # (C_out, K)
    
    y_2d = torch.empty((M, C_out), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    
    conv1x1_bias_relu_kernel[grid](
        x_2d, w_2d, bias, y_2d,
        M, C_out, K,
        x_2d.stride(0), x_2d.stride(1),
        w_2d.stride(0), w_2d.stride(1),
        y_2d.stride(0), y_2d.stride(1),
        APPLY_RELU=apply_relu,
    )
    
    # Reshape back to (N, C_out, H, W)
    return y_2d.reshape(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


# ============================================================
# Triton kernel: fused conv2d (3x3, pad=1) + bias + relu
# Implicit GEMM approach
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K', 'IN_H', 'IN_W', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv3x3_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    M, N, K,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    W: (OUT_C, IN_C, KH, KW) -> reshaped to (OUT_C, K) where K=IN_C*KH*KW
    M = BATCH * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KERNEL_H * KERNEL_W
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K = IN_C * KERNEL_H * KERNEL_W
    # Iterate over K in blocks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c, kh, kw)
        kw_idx = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh_idx = tmp_k % KERNEL_H
        in_c_idx = tmp_k // KERNEL_H
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_h = (ih >= 0) & (ih < IN_H)
        valid_w = (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx[:, None] < BATCH
        
        x_mask = valid_m & valid_k & valid_h & valid_w & valid_batch
        
        # X in NHWC: (N, H, W, C) -> stride: (IN_H*IN_W*IN_C, IN_W*IN_C, IN_C, 1)
        x_ptrs = X_ptr + (
            batch_idx[:, None] * (IN_H * IN_W * IN_C) +
            in_c_idx[None, :]
        )
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # W reshaped to (OUT_C, K): stride (K, 1)
        # But we need (K, OUT_C) for the dot product
        # W_ptr is (OUT_C, K), we load (BLOCK_K, BLOCK_N)
        w_ptrs = W_ptr + offs_k[:, None] + offs_n[None, :] * K
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)
    
    # Store to Y in NHWC: (N, H, W, C)
    y_ptrs = Y_ptr + (
        batch_idx[:, None] * (OUT_H * OUT_W * OUT_C) +
        out_h_idx[:, None] * (OUT_W * OUT_C) +
        out_w_idx[:, None] * OUT_C +
        offs_n[None, :]
    )
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.float32), mask=y_mask)


def conv3x3_bias_relu(x_nhwc, weight, bias, stride=1, padding=1, apply_relu=True):
    """
    returns: (N, H, W, C_out) NHWC
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    C_out, _, KH, KW = weight.shape
    
    OUT_H = (IN_H + 2 * padding - KH) // stride + 1
    OUT_W = (IN_W + 2 * padding - KW) // stride + 1
    
    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW
    
    # Reshape weight to (OUT_C, K) then transpose for our kernel: we store as (OUT_C, K)
    # In kernel we access W_ptr + offs_k[:, None] + offs_n[None, :] * K
    # So W is stored as (OUT_C, K) with stride (K, 1), accessed as w[n, k] = W_ptr[n*K + k]
    # We need w_tile[k, n] so we load w_ptrs = W_ptr + offs_k[:, None] + offs_n[None, :] * K
    # This gives w[k, n] = W_ptr[offs_n*K + offs_k] which is W[n, k] - correct!
    w_2d = weight.reshape(C_out, K).contiguous()
    
    y_nhwc = torch.empty((N, OUT_H, OUT_W, C_out), device=x_nhwc.device, dtype=x_nhwc.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    
    conv3x3_bias_relu_kernel[grid](
        x_nhwc, w_2d, bias, y_nhwc,
        N, IN_C, IN_H, IN_W,
        C_out, OUT_H, OUT_W,
        M, C_out, K,
        stride, stride,
        padding, padding,
        KH, KW,
        APPLY_RELU=apply_relu,
    )
    
    return y_nhwc


# ============================================================
# Triton kernel: max_pool2d (3x3, stride=2, ceil_mode=True)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['total'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    """
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    
    # Decode: total = BATCH * OUT_H * OUT_W * C
    c_idx = offs % C
    tmp = offs // C
    ow_idx = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh_idx = tmp2 % OUT_H
    n_idx = tmp2 // OUT_H
    
    # Compute max over kernel window
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            ih = oh_idx * STRIDE_H + kh - PAD_H
            iw = ow_idx * STRIDE_W + kw - PAD_W
            
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (n_idx < BATCH)
            
            x_ptrs = X_ptr + n_idx * (IN_H * IN_W * C) + ih * (IN_W * C) + iw * C + c_idx
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    y_ptrs = Y_ptr + n_idx * (OUT_H * OUT_W * C) + oh_idx * (OUT_W * C) + ow_idx * C + c_idx
    tl.store(y_ptrs, max_val, mask=mask)


def maxpool2d_nhwc(x_nhwc, kernel_size=3, stride=2, padding=0, ceil_mode=True):
    """
    returns: (N, OUT_H, OUT_W, C) NHWC
    """
    N, IN_H, IN_W, C = x_nhwc.shape
    
    if ceil_mode:
        OUT_H = math.ceil((IN_H + 2 * padding - kernel_size) / stride) + 1
        OUT_W = math.ceil((IN_W + 2 * padding - kernel_size) / stride) + 1
    else:
        OUT_H = (IN_H + 2 * padding - kernel_size) // stride + 1
        OUT_W = (IN_W + 2 * padding - kernel_size) // stride + 1
    
    y_nhwc = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=x_nhwc.dtype)
    total = N * OUT_H * OUT_W * C
    
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    
    maxpool2d_kernel[grid](
        x_nhwc, y_nhwc,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        total,
    )
    
    return y_nhwc


# ============================================================
# Triton kernel: adaptive_avg_pool2d to (1,1)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    X_ptr, Y_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N * C
    
    c_idx = offs % C
    n_idx = offs // C
    
    # Sum over H*W
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    total_pixels = H * W
    
    for h in range(H):
        for w in range(W):
            x_ptrs = X_ptr + n_idx * (H * W * C) + h * (W * C) + w * C + c_idx
            val = tl.load(x_ptrs, mask=mask, other=0.0)
            acc = acc + val.to(tl.float32)
    
    avg = acc / total_pixels
    
    y_ptrs = Y_ptr + n_idx * C + c_idx
    tl.store(y_ptrs, avg.to(tl.float32), mask=mask)


def adaptive_avg_pool2d_1x1_nhwc(x_nhwc):
    """
    returns: (N, C) - equivalent to (N, C, 1, 1) flattened
    """
    N, H, W, C = x_nhwc.shape
    y = torch.empty((N, C), device=x_nhwc.device, dtype=x_nhwc.dtype)
    
    grid = lambda meta: (triton.cdiv(N * C, meta['BLOCK_SIZE']),)
    
    adaptive_avg_pool2d_1x1_kernel[grid](
        x_nhwc, y,
        N, C, H, W,
    )
    
    return y


# ============================================================
# Large conv (7x7, stride=2) kernel
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_general_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    M, N, K,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        kw_idx = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh_idx = tmp_k % KERNEL_H
        in_c_idx = tmp_k // KERNEL_H
        
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_h = (ih >= 0) & (ih < IN_H)
        valid_w = (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx[:, None] < BATCH
        
        x_mask = valid_m & valid_k & valid_h & valid_w & valid_batch
        
        x_ptrs = X_ptr + (
            batch_idx[:, None] * (IN_H * IN_W * IN_C) +
            in_c_idx[None, :]
        )
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = W_ptr + offs_k[:, None] + offs_n[None, :] * K
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)
    
    y_ptrs = Y_ptr + (
        batch_idx[:, None] * (OUT_H * OUT_W * OUT_C) +
        out_h_idx[:, None] * (OUT_W * OUT_C) +
        out_w_idx[:, None] * OUT_C +
        offs_n[None, :]
    )
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.float32), mask=y_mask)


def conv_general_bias_relu(x_nhwc, weight, bias, stride=1, padding=0, apply_relu=True):
    """
    returns: (N, OUT_H, OUT_W, C_out) NHWC
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    C_out, _, KH, KW = weight.shape
    
    OUT_H = (IN_H + 2 * padding - KH) // stride + 1
    OUT_W = (IN_W + 2 * padding - KW) // stride + 1
    
    M = N * OUT_H * OUT