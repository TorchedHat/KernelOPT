import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Fused BatchNorm + ReLU kernel (inference mode)
# ─────────────────────────────────────────────────────────────────────────────
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
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Fused BatchNorm (no relu) kernel
# ─────────────────────────────────────────────────────────────────────────────
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
def bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    tl.store(out_ptr + offsets, out, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Fused BatchNorm + Add + ReLU kernel
# ─────────────────────────────────────────────────────────────────────────────
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
def bn_add_relu_kernel(
    x_ptr, residual_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b + res
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Average Pool 2D (global) + flatten kernel
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=2),
    ],
    key=['HW', 'NC'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    x_ptr, out_ptr,
    N, C, HW, NC,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (n, c) pair
    nc_idx = tl.program_id(0)
    n = nc_idx // C
    c = nc_idx % C
    
    base = n * C * HW + c * HW
    
    acc = tl.zeros([1], dtype=tl.float32)
    for hw_off in range(0, HW, BLOCK_HW):
        hw_offsets = hw_off + tl.arange(0, BLOCK_HW)
        mask = hw_offsets < HW
        vals = tl.load(x_ptr + base + hw_offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)
    
    out = acc / HW
    tl.store(out_ptr + nc_idx, out)


# ─────────────────────────────────────────────────────────────────────────────
# Linear kernel (matmul: [N, K] x [K, M] -> [N, M])
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # w is [N, K], we want [K, N] for matmul
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Channel Shuffle kernel
# ─────────────────────────────────────────────────────────────────────────────
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
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    numel,
    N, C, HW, groups,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Decode output index: out[n, c_out, hw] where c_out = g * cpg + cpg_idx
    # Input layout: x[n, g, cpg, hw] -> out[n, cpg, g, hw]
    # out index: n * C * HW + c_out * HW + hw
    # c_out = cpg_idx * groups + g
    # input index: n * C * HW + (g * cpg + cpg_idx) * HW + hw
    
    hw = offsets % HW
    tmp = offsets // HW
    c_out = tmp % C
    n = tmp // C
    
    cpg = C // groups
    g = c_out % groups
    cpg_idx = c_out // groups
    
    c_in = g * cpg + cpg_idx
    
    in_offsets = n * C * HW + c_in * HW + hw
    
    vals = tl.load(x_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, vals, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def apply_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * HW
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, HW,
        eps,
    )
    return out


def apply_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * HW
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, HW,
        eps,
    )
    return out


def apply_bn_add_relu(x, residual, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * HW
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_add_relu_kernel[grid](
        x, residual, out,
        running_mean, running_var, weight, bias,
        numel, C, HW,
        eps,
    )
    return out


def apply_channel_shuffle(x, groups):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * HW
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid](
        x, out,
        numel,
        N, C, HW, groups,
    )
    return out


def apply_conv2d(x, weight, stride=1, padding=0, groups=1):
    """Use PyTorch's conv2d - but we need to implement in Triton per rules.
    # Actually per the rules, conv2d IS banned. We need to implement it.
    # But the rules also say "KEEP the matmul/conv as a standard PyTorch call"
    # for performance. However the anti-cheating says F.conv2d is banned.
    # We'll implement a simple conv2d using unfold + matmul approach via Triton.
    # For correctness and given the complexity, let's use the implicit GEMM kernel.
    return _conv2d_triton(x, weight, stride, padding, groups)


def _conv2d_triton(x, weight, stride=1, padding=0, groups=1):
    """Implicit GEMM convolution using Triton."""
    N, C_in, H, W = x.shape
    C_out, C_in_per_group, kH, kW = weight.shape
    
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    out_H = (H + 2 * pad_h - kH) // stride_h + 1
    out_W = (W + 2 * pad_w - kW) // stride_w + 1
    
    out = torch.zeros(N, C_out, out_H, out_W, device=x.device, dtype=x.dtype)
    
    # Use unfold for simplicity and correctness
    # x_unfold: [N, C_in * kH * kW, out_H * out_W]
    x_unfold = torch.nn.functional.unfold(x, (kH, kW), padding=(pad_h, pad_w), stride=(stride_h, stride_w))
    # x_unfold: [N, C_in*kH*kW, out_H*out_W]
    
    # weight: [C_out, C_in_per_group, kH, kW] -> [C_out, C_in_per_group*kH*kW]
    w_flat = weight.view(C_out, -1)
    
    if groups == 1:
        # [N, C_out, out_H*out_W] = [C_out, K] x [N, K, out_H*out_W]
        # Use our linear kernel
        # x_unfold: [N, K, L] where K = C_in*kH*kW, L = out_H*out_W
        # We want: out[n, co, l] = sum_k w_flat[co, k] * x_unfold[n, k, l]
        # Reshape to [N*L, K] x [K, C_out] -> [N*L, C_out]
        L = out_H * out_W
        K = C_in * kH * kW
        x_2d = x_unfold.permute(0, 2, 1).contiguous().view(N * L, K)  # [N*L, K]
        w_2d = w_flat.t().contiguous()  # [K, C_out]
        
        out_2d = _matmul_triton(x_2d, w_2d)  # [N*L, C_out]
        out = out_2d.view(N, L, C_out).permute(0, 2, 1).contiguous().view(N, C_out, out_H, out_W)
    else:
        # Group convolution
        group_in = C_in // groups
        group_out = C_out // groups
        K_group = group_in * kH * kW
        L = out_H * out_W
        
        out_parts = []
        for g in range(groups):
            x_g = x_unfold[:, g * K_group:(g + 1) * K_group, :]  # [N, K_group, L]
            w_g = w_flat[g * group_out:(g + 1) * group_out, :]    # [group_out, K_group]
            
            x_2d = x_g.permute(0, 2, 1).contiguous().view(N * L, K_group)
            w_2d = w_g.t().contiguous()
            
            out_2d = _matmul_triton(x_2d, w_2d)
            out_parts.append(out_2d.view(N, L, group_out).permute(0, 2, 1).contiguous().view(N, group_out, out_H, out_W))
        
        out = torch.cat(out_parts, dim=1)
    
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc = tl.dot(a_tile, b_tile, acc=acc)
    
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _matmul_triton(a, b):
    """a: [M, K], b: [K, N] -> [M, N]"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
    )
    return c


def apply_adaptive_avg_pool(x):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    grid = (NC,)
    adaptive_avg_pool_kernel[grid](
        x, out.view(-1),
        N, C, HW, NC,
    )
    return out


def apply_linear(x, weight, bias):
    """x: [M, K], weight: [N, K], bias: [N] -> [M, N]"""
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ShuffleNet block
# ─────────────────────────────────────────────────────────────────────────────
def shufflenet_unit(x, 
                    conv1_w, bn1_w, bn1_b, bn1_mean, bn1_var,
                    conv2_w, bn2_w, bn2_b, bn2_mean, bn2_var,
                    conv3_w, bn3_w, bn3_b, bn3_mean, bn3_var,
                    shortcut_conv_w=None, shortcut_bn_w=None, shortcut_bn_b=None,
                    shortcut_bn_mean=None, shortcut_bn_var=None,
                    groups=3):
    """One ShuffleNet unit."""
    # conv1: 1x1 group conv
    out = apply_conv2d(x, conv1_w, stride=1, padding=0, groups=groups)
    out = apply_bn_relu(out, bn1_w, bn1_b, bn1_mean, bn1_var)
    
    # conv2: depthwise 3x3
    mid_c = out.shape[1]
    out = apply_conv2d(out, conv2_w, stride=1, padding=1, groups=mid_c)
    out = apply_bn(out, bn2_w, bn2_b, bn2_mean, bn2_var)
    
    # channel shuffle
    out = apply_channel_shuffle(out, groups)
    
    # conv3: 1x1 group conv
    out = apply_conv2d(out, conv3_w, stride=1, padding=0, groups=groups)
    
    # shortcut
    if shortcut_conv_w is not None:
        sc = apply_conv2d(x, shortcut_conv_w, stride=1, padding=0, groups=1)
        sc = apply_bn(sc, shortcut_bn_w, shortcut_bn_b, shortcut_bn_mean, shortcut_bn_var)
    else:
        sc = x
    
    out = apply_bn_add_relu(out, sc, bn3_w, bn3_b, bn3_mean, bn3_var)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel function
# ─────────────────────────────────────────────────────────────────────────────
def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_stage2_0_conv1_weight, p_stage2_0_bn1_weight, p_stage2_0_bn1_bias,
    p_stage2_0_conv2_weight, p_stage2_0_bn2_weight, p_stage2_0_bn2_bias,
    p_stage2_0_conv3_weight, p_stage2_0_bn3_weight, p_stage2_0_bn3_bias,
    p_stage2_0_shortcut_0_weight, p_stage2_0_shortcut_1_weight, p_stage2_0_shortcut_1_bias,
    p_stage2_1_conv1_weight, p_stage2_1_bn1_weight, p_stage2_1_bn1_bias,
    p_stage2_1_conv2_weight, p_stage2_1_bn2_weight, p_stage2_1_bn2_bias,
    p_stage2_1_conv3_weight, p_stage2_1_bn3_weight, p_stage2_1_bn3_bias,
    p_stage2_2_conv1_weight, p_stage2_2_bn1_weight, p_stage2_2_bn1_bias,
    p_stage2_2_conv2_weight, p_stage2_2_bn2_weight, p_stage2_2_bn2_bias,
    p_stage2_2_conv3_weight, p_stage2_2_bn3_weight, p_stage2_2_bn3_bias,
    p_stage3_0_conv1_weight, p_stage3_0_bn1_weight, p_stage3_0_bn1_bias,