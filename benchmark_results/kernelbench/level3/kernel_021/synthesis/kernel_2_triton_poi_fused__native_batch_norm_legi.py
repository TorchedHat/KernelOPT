import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ============================================================
# Kernel 1: BN + ReLU6 pointwise fusion
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu6_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C,
    eps,
    do_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # channel index: layout is (N, C, H, W) so channel = (offset // (H*W)) % C
    # But we pass C and total = N*C*H*W, spatial = H*W
    # channel index = (offset // spatial) % C, but we don't have spatial directly
    # Instead: channel = (offset // (numel // (N*C))) % C
    # Simpler: we know layout is NCHW, so channel = (offset / spatial) % C
    # We'll compute it as: channel = (offset % (C * spatial)) // spatial
    # Actually just use: c_idx = (offset // 1) % C won't work for NCHW
    # For NCHW with spatial S: channel = (offset // S) % C
    # We pass spatial as a parameter
    # Let's just use offset % C for channels-last, but output is NCHW
    # For NCHW: index = n * C * H * W + c * H * W + h * W + w
    # So c = (index // (H*W)) % C
    # We need H*W = numel / (N*C) -- but we don't have N separately
    # Pass spatial = H*W
    c_idx = (offsets // 1) % C  # placeholder, will fix with spatial param
    
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# This approach won't work well because we need spatial. Let me redesign.

# ============================================================
# Better approach: separate kernels with proper indexing
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def bn_relu6_nchw_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, spatial,
    eps: tl.constexpr,
    do_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    c_idx = (offsets // spatial) % (numel // spatial)  # won't work without C
    # Need C explicitly
    tl.store(out_ptr + offsets, x, mask=mask)


# Let me write the actual correct kernels

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'C', 'spatial'],
)
@triton.jit
def fused_bn_relu6_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, spatial,
    do_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    c_idx = (offsets // spatial) % C
    
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    
    eps = 1e-5
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Expand conv: 1x1 conv (N,112,224,224) x (672,112,1,1) -> (N,672,224,224)
# This is a GEMM: M=N*H*W=10*224*224=501760, K=112, N_out=672
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_gemm_kernel(
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
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                     mask=x_mask, other=0.0)
        w = tl.load(w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk,
                     mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc, allow_tf32=False)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=out_mask)


# ============================================================
# Depthwise conv 5x5, stride=2, padding=2
# Input: (N, C, H, W) = (10, 672, 224, 224)
# Output: (N, C, H//2, W//2) = (10, 672, 112, 112)
# Weight: (C, 1, 5, 5) = (672, 1, 5, 5)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32, 'BLOCK_C': 4}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 4}, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_C': 2}, num_warps=4),
        triton.Config({'BLOCK_N': 16, 'BLOCK_C': 8}, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_C': 8}, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 8}, num_warps=8),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def depthwise_conv5x5_s2_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # Grid: (OUT_H * OUT_W, cdiv(N*C, BLOCK_N*BLOCK_C))
    # Actually let's do: pid_spatial covers output spatial positions
    # pid_nc covers (batch, channel) pairs
    
    pid_spatial = tl.program_id(0)
    pid_nc = tl.program_id(1)
    
    oh = pid_spatial // OUT_W
    ow = pid_spatial % OUT_W
    
    nc_base = pid_nc * BLOCK_N * BLOCK_C
    
    for nc_off in range(0, BLOCK_N * BLOCK_C):
        nc_idx = nc_base + nc_off
        if nc_idx < N * C:
            n_idx = nc_idx // C
            c_idx = nc_idx % C
            
            acc = tl.zeros((), dtype=tl.float32)
            for kh in range(KH):
                for kw in range(KW):
                    ih = oh * STRIDE_H + kh - PAD_H
                    iw = ow * STRIDE_W + kw - PAD_W
                    if ih >= 0 and ih < IN_H and iw >= 0 and iw < IN_W:
                        x_val = tl.load(x_ptr + n_idx * C * IN_H * IN_W + c_idx * IN_H * IN_W + ih * IN_W + iw)
                        w_val = tl.load(w_ptr + c_idx * KH * KW + kh * KW + kw)
                        acc += x_val * w_val
            
            tl.store(out_ptr + n_idx * C * OUT_H * OUT_W + c_idx * OUT_H * OUT_W + oh * OUT_W + ow, acc)