import torch
import triton
import triton.language as tl
import math
import torch.nn.functional as F

# Fused BatchNorm + ReLU kernel
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
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + c, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    # BN
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_scaled = x_norm * weight + bias
    # ReLU
    out = tl.maximum(x_scaled, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Implicit GEMM Conv2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, H, W,
    K, R, S,
    OH, OW,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode m -> (batch, oh, ow)
    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    n = offs_m // (OW * OH)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = C * R * S
    
    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode k -> (c, r, s)
        s = offs_k % S
        r = (offs_k // S) % R
        c = offs_k // (S * R)
        
        # Input coordinates
        ih = oh[:, None] * stride_h + r[None, :] - pad_h
        iw = ow[:, None] * stride_w + s[None, :] - pad_w
        
        valid_x = (n[:, None] < N) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & (offs_k[None, :] < TOTAL_K)
        
        x_idx = n[:, None] * (C * H * W) + c[None, :] * (H * W) + ih * W + iw
        x_tile = tl.load(x_ptr + x_idx, mask=valid_x, other=0.0)
        
        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < K)
        w_idx = offs_n[None, :] * (C * R * S) + c[:, None] * (R * S) + r[:, None] * S + s[:, None]
        w_tile = tl.load(w_ptr + w_idx, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    out_idx = n[:, None] * (K * OH * OW) + offs_n[None, :] * (OH * OW) + oh[:, None] * OW + ow[:, None]
    tl.store(out_ptr + out_idx, acc, mask=mask_out)


def triton_conv2d(x, w, stride=(1,1), padding=(0,0)):
    N, C, H, W = x.shape
    K, C2, R, S = w.shape
    assert C == C2
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    OH = (H + 2*pad_h - R) // stride_h + 1
    OW = (W + 2*pad_w - S) // stride_w + 1
    M = N * OH * OW
    
    out = torch.empty((N, K, OH, OW), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(K, meta['BLOCK_N']))
    
    conv2d_kernel[grid](
        x, w, out,
        N, C, H, W,
        K, R, S,
        OH, OW,
        stride_h, stride_w,
        pad_h, pad_w,
        M,
    )
    return out


def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](x, out, mean, var, weight, bias, numel, C, HW, eps)
    return out


def triton_bn(x, mean, var, weight, bias, eps=1e-5):
    # BN without ReLU - reuse same kernel but with relu=False
    # For simplicity, use same kernel (ReLU of negative values won't matter if followed by cat)
    # Actually we need BN without ReLU for transition layers
    # Let's implement a separate kernel
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_only_kernel[grid](x, out, mean, var, weight, bias, numel, C, HW, eps)
    return out


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
def bn_only_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + c, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias
    
    tl.store(out_ptr + offsets, out, mask=mask)