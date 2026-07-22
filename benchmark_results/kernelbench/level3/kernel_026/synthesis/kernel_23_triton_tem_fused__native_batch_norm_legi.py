import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
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
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def fused_bn_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = (offsets // HW) % C
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def fused_bn_add_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    residual_ptr,
    out_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    c = (offsets // HW) % C
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b + res
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    offs_m_out = offs_m
    offs_n_out = offs_n
    
    b = tl.load(b_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
    acc = acc + b[None, :]
    
    out_mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)
    tl.store(out_ptr + offs_m_out[:, None] * N + offs_n_out[None, :], acc, mask=out_mask)


def apply_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, mean, var, weight, bias, out,
    )
    return out


def apply_bn(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_kernel[grid](
        x, mean, var, weight, bias, out,
    )
    return out


def apply_bn_add_relu(x, mean, var, weight, bias, residual, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_add_relu_kernel[grid](
        x, mean, var, weight, bias, residual, out,
    )
    return out


def channel_shuffle(x, groups):
    N, C, H, W = x.shape
    cpg = C // groups
    x = x.view(N, groups, cpg, H, W)
    x = x.transpose(1, 2).contiguous()
    x = x.view(N, C, H, W)
    return x


def conv2d_op(x, weight, stride=1, padding=0, groups=1):
    # Use PyTorch's conv2d - but we need to implement in Triton per rules
    # However, for correctness and since conv is the dominant op, we implement
    # a simple wrapper that calls the underlying CUDA conv
    # Per the rules, we cannot use F.conv2d, so we implement via Triton implicit GEMM
    # But given the complexity, let's use the allowed approach:
    # The rules say we cannot use F.conv2d. We need Triton conv.
    # For this complex model, let's implement a Triton conv2d
    pass


# Since implementing full Triton conv2d for all cases is very complex,
# and the rules say we must implement in Triton, let me implement the 
# full forward pass using Triton conv kernels.

# Actually, re-reading the rules: "Banned convolution delegation: F.conv2d..."
# But also: "KEEP the matmul/conv as a standard PyTorch call"
# These are contradictory. The "ANTI-CHEATING" section bans F.conv2d.
# But the "CRITICAL" section says to keep conv as PyTorch call.
# The anti-cheating section takes precedence based on "ANTI-CHEATING CONSTRAINTS" header.

# Let me implement Triton conv2d kernels for the key operations.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wk,
    OUT_H, OUT_W,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 1x1 conv is just a matrix multiply
    # M = batch * out_h * out_w, N = out_c, K = in_c
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # decode m -> (batch, h, w)
    w_idx = offs_m % OUT_W
    h_idx = (offs_m // OUT_W) % OUT_H
    n_idx = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        
        x_ptrs = x_ptr + n_idx[:, None] * stride_xn + k_offs[None, :] * stride_xc + h_idx[:, None] * stride_xh + w_idx[:, None] * stride_xw
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # output in NCHW: out[n, c, h, w] = acc
    # linear index: n * N * OUT_H * OUT_W + c * OUT_H * OUT_W + h * OUT_W + w
    out_idx = n_idx[:, None] * (N * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + h_idx[:, None] * OUT_W + w_idx[:, None]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K', 'GROUPS'],
)
@triton.jit
def conv2d_1x1_grouped_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    BATCH, IN_C, OUT_C, OUT_H, OUT_W,
    GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grouped 1x1 conv
    # Each group handles IN_C//GROUPS input channels -> OUT_C//GROUPS output channels
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)
    
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    w_idx = offs_m % OUT_W
    h_idx = (offs_m // OUT_W) % OUT_H
    n_idx = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, GROUP_IN_C, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        global_k = pid_g * GROUP_IN_C + k_offs
        
        x_ptrs = x_ptr + n_idx[:, None] * (IN_C * OUT_H * OUT_W) + global_k[None, :] * (OUT_H * OUT_W) + h_idx[:, None] * OUT_W + w_idx[:, None]
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < GROUP_IN_C)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        global_n = pid_g * GROUP_OUT_C + offs_n
        w_ptrs = w_ptr + global_n[None, :] * GROUP_IN_C + k_offs[:, None]
        w_mask = (offs_n[None, :] < GROUP_OUT_C) & (k_offs[:, None] < GROUP_IN_C)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    global_n_out = pid_g * GROUP_OUT_C + offs_n
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < GROUP_OUT_C) & (global_n_out[None, :] < OUT_C)
    out_idx = n_idx[:, None] * (OUT_C * OUT_H * OUT_W) + global_n_out[None, :] * (OUT_H * OUT_W) + h_idx[:, None] * OUT_W + w_idx[:, None]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 1, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 1, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 1, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 1, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M', 'C'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    M,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Depthwise 3x3 conv with padding=1, stride=1
    # Each output element: out[n,c,h,w] = sum_{kh,kw} x[n,c,h+kh-1,w+kw-1] * w[c,kh,kw]
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M
    
    w_idx = offs % OUT_W
    h_idx = (offs // OUT_W) % OUT_H
    c_idx = (offs // (OUT_W * OUT_H)) % C
    n_idx = offs // (OUT_W * OUT_H * C)
    
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # 3x3 kernel unrolled
    for kh in range(3):
        for kw in range(3):
            ih = h_idx + kh - 1
            iw = w_idx + kw - 1
            valid = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & mask
            x_idx = n_idx * (C * IN_H * IN_W) + c_idx * (IN_H * IN_W) + ih * IN_W + iw
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            w_idx2 = c_idx * 9 + kh * 3 + kw
            w_val = tl.load(w_ptr + w_idx2, mask=mask, other=0.0)
            acc += x_val * w_val
    
    tl.store(out_ptr + offs, acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K', 'STRIDE_H', 'STRIDE_W'],
)
@triton.jit
def conv2d_3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    BATCH, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
    PAD_H, PAD_W,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    w_idx = offs_m % OUT_W
    h_idx = (offs_m // OUT_W) % OUT_H
    n_idx = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for kh in range(3):
        for kw in range(3):
            ih = h_idx * STRIDE_H + kh - PAD_H
            iw = w_idx * STRIDE_W + kw - PAD_W
            
            for k in range(0, K, BLOCK_K):
                k_offs = k + tl.arange(0, BLOCK_K)
                
                valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
                x_ptrs = x_ptr + n_idx[:, None] * (IN_C * IN_H * IN_W) + k_offs[None, :] * (IN_H * IN_W) + ih[:, None] * IN_W + iw[:, None]
                x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K) & valid_spatial[:, None]
                x = tl.load(x_ptrs, mask=x_mask, other=0.0)
                
                w_ptrs = w_ptr + offs_n[None, :] * (K * 9) + k_offs[:, None] * 9 + kh * 3 + kw
                w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
                w = tl.load(w_ptrs, mask=w_mask, other=0.0)
                
                acc = tl.dot(x, w, acc)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_idx = n_idx[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + h_idx[:, None] * OUT_W + w_idx[:, None]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def triton_conv2d_1x1(x, weight, groups=1):
    """1x1 convolution using Triton"""
    N, IN_C, H, W = x.shape
    OUT_C = weight.shape[0]
    
    out = torch.empty(N, OUT_C, H, W, device=x.device, dtype=x.dtype)
    M = N * H * W
    
    if groups == 1:
        K = IN_C
        # weight shape: [OUT_C, IN_C, 1, 1] -> [OUT_C, IN_C]
        w_flat = weight.view(OUT_C, IN_C)
        
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
        conv2d_1x1_kernel[grid](
            x, w_flat, out,
            M, OUT_C, K,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w_flat.stride(0), w_flat.stride(1),
            H, W,
        )
    else:
        GROUP_IN_C = IN_C // groups
        GROUP_OUT_C = OUT_C // groups
        # weight shape: [OUT_C, IN_C//groups, 1, 1] -> [OUT_C, GROUP_IN_C]
        w_flat = weight.view(OUT_C, GROUP_IN_C)
        
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(GROUP_OUT_C, meta['BLOCK_N']), groups)
        conv2d_1x1_grouped_kernel[grid](
            x, w_flat, out,
            M, OUT_C, GROUP_IN_C,
            N, IN_C, OUT_C, H, W,
            groups,
        )
    
    return out


def triton_depthwise_conv3x3(x, weight):
    """Depthwise 3x3 conv with padding=1, stride=1"""
    N, C, H, W = x.shape
    out = torch.empty(N, C, H, W, device=x.device, dtype=x.dtype)
    M = N * C * H * W
    
    # weight shape: [C, 1, 3, 3]
    w_flat = weight.view(C, 9)
    # Need contiguous weight indexed as [c, kh*3+kw]
    w_cont = w_flat.contiguous()
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    depthwise_conv3x3_kernel[grid](
        x, w_cont, out,
        N, C, H, W, H, W,
        M,
    )
    return out


def triton_conv2d_3x3(x, weight, stride=1, padding=1):
    """General 3x3 convolution"""
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    OUT_H = (IN_H + 2 * padding - KH) // stride + 1
    OUT_W = (IN_W + 2 * padding - KW) // stride + 1
    
    out = torch.empty(N, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    M = N * OUT_H * OUT_W
    K = IN_C
    
    # weight: [OUT_C, IN_C, 3, 3]
    grid =