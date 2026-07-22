import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused BatchNorm + ReLU pointwise kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def relu_kernel(
    x_ptr, out_ptr, numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def sigmoid_kernel(
    x_ptr, out_ptr, numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# Adaptive avg pool 2d -> (N, C, 1, 1)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 32}, num_warps=2),
    ],
    key=['HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    if n >= N:
        return
    
    base = n * C * HW + c * HW
    acc = tl.zeros([1], dtype=tl.float32)
    
    for hw_off in range(0, HW, BLOCK_HW):
        hw_idx = hw_off + tl.arange(0, BLOCK_HW)
        mask = hw_idx < HW
        x = tl.load(x_ptr + base + hw_idx, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    
    out = acc / HW
    tl.store(out_ptr + n * C + c, out)


# Conv2d 1x1 (pointwise) using GEMM
# Input: (N, C_in, H, W), Weight: (C_out, C_in, 1, 1)
# Treat as GEMM: (N*H*W, C_in) x (C_in, C_out) -> (N*H*W, C_out)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # w shape: (C_out, C_in) -> we need (K, N) = (C_in, C_out)
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x, w, acc=acc, allow_tf32=False)
    
    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_out = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)
    out_ptrs = out_ptr + offs_m_out[:, None] * stride_om + offs_n_out[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# Conv2d 3x3 depthwise
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_NC': 128, 'BLOCK_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_NC': 64, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_NC': 256, 'BLOCK_HW': 16}, num_warps=8),
        triton.Config({'BLOCK_NC': 32, 'BLOCK_HW': 128}, num_warps=4),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BLOCK_NC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    total_nc = N * C
    total_hw = OUT_H * OUT_W
    
    nc_idx = (pid * BLOCK_NC) // (BLOCK_NC)  # This won't work well
    # Better: use 2D grid
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    nc_offs = pid_nc * BLOCK_NC + tl.arange(0, BLOCK_NC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    
    nc_mask = nc_offs < total_nc
    hw_mask = hw_offs < total_hw
    
    n = nc_offs // C
    c = nc_offs % C
    
    out_h = hw_offs // OUT_W
    out_w = hw_offs % OUT_W
    
    # For each (nc, hw) pair, compute the depthwise conv
    # This is tricky to vectorize well; let's do it element by element
    for nc_i in range(BLOCK_NC):
        nc_val = pid_nc * BLOCK_NC + nc_i
        if nc_val >= total_nc:
            break
        n_val = nc_val // C
        c_val = nc_val % C
        
        for hw_i in range(BLOCK_HW):
            hw_val = pid_hw * BLOCK_HW + hw_i
            if hw_val >= total_hw:
                break
            oh = hw_val // OUT_W
            ow = hw_val % OUT_W
            
            acc = 0.0
            for kh in range(3):
                for kw in range(3):
                    ih = oh * STRIDE_H + kh - PAD_H
                    iw = ow * STRIDE_W + kw - PAD_W
                    if ih >= 0 and ih < IN_H and iw >= 0 and iw < IN_W:
                        x_val = tl.load(x_ptr + n_val * C * IN_H * IN_W + c_val * IN_H * IN_W + ih * IN_W + iw)
                        w_val = tl.load(w_ptr + c_val * 9 + kh * 3 + kw)
                        acc += x_val * w_val
            
            tl.store(out_ptr + n_val * C * OUT_H * OUT_W + c_val * OUT_H * OUT_W + oh * OUT_W + ow, acc)