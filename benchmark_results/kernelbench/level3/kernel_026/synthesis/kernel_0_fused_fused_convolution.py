import torch
import triton
import triton.language as tl
import math


import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernel: fused BatchNorm (inference) + optional ReLU
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
    weight_ptr, bias_ptr, mean_ptr, var_ptr,
    numel, C, HW,
    eps,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    # Layout: N, C, H, W -> flat index = n*C*HW + c*HW + hw
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu:
        out = tl.maximum(out, tl.zeros_like(out))
    
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Triton kernel: channel shuffle
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    groups,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # out[n, c, h, w] = in[n, c', h, w]
    # where c' is the shuffled channel index
    # Original: view as (N, groups, C//groups, H, W), transpose(1,2) -> (N, C//groups, groups, H, W), view as (N, C, H, W)
    # So: out_c = (in_c % groups) * (C // groups) + (in_c // groups)
    # Equivalently: for output index c, the input channel is:
    # out[n, c, h, w] comes from in[n, (c % (C//groups)) * groups + (c // (C//groups)), h, w]
    
    HW = H * W
    CHW = C * HW
    
    n = offsets // CHW
    rem = offsets % CHW
    c_out = rem // HW
    hw = rem % HW
    
    channels_per_group = C // groups
    # c_out = g_out * channels_per_group + cpg_out
    # This came from transposing: original was (groups, channels_per_group)
    # After transpose: (channels_per_group, groups)
    # So c_out index in (cpg, g) space: cpg = c_out // groups, g = c_out % groups
    # Original c_in in (g, cpg) space: c_in = g * channels_per_group + cpg
    cpg = c_out // groups
    g = c_out % groups
    c_in = g * channels_per_group + cpg
    
    in_offsets = n * CHW + c_in * HW + hw
    
    val = tl.load(x_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


# ============================================================
# Triton kernel: max pool 2d (3x3, stride=2, padding=1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    OHW = OUT_H * OUT_W
    OCHW = C * OHW
    
    n = offsets // OCHW
    rem = offsets % OCHW
    c = rem // OHW
    rem2 = rem % OHW
    oh = rem2 // OUT_W
    ow = rem2 % OUT_W
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SW + kh - PH  # Note: using SW for SH here, fix below
            iw = ow * SW + kw - PW
            ih_real = oh * SH + kh - PH
            iw_real = ow * SW + kw - PW
            valid = mask & (ih_real >= 0) & (ih_real < IN_H) & (iw_real >= 0) & (iw_real < IN_W) & (n < N)
            in_idx = n * C * IN_H * IN_W + c * IN_H * IN_W + ih_real * IN_W + iw_real
            val = tl.load(x_ptr + in_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


# ============================================================
# Triton kernel: adaptive avg pool 2d -> (1,1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['NC', 'HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    x_ptr, out_ptr,
    NC, HW,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    nc_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    nc_mask = nc_offsets < NC
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over HW
    for hw in range(HW):
        in_idx = nc_offsets * HW + hw
        val = tl.load(x_ptr + in_idx, mask=nc_mask, other=0.0)
        acc += val
    
    avg = acc / HW
    tl.store(out_ptr + nc_offsets, avg, mask=nc_mask)


# ============================================================
# Triton kernel: linear (matrix-vector or matrix-matrix)
# Using tl.dot for the actual computation
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        
        x_tile = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, tl.trans(w_tile), acc)
    
    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# ============================================================
# Triton kernel: fused BN + add (residual) + optional ReLU
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
def bn_add_relu_kernel(
    x_ptr, residual_ptr, out_ptr,
    weight_ptr, bias_ptr, mean_ptr, var_ptr,
    numel, C, HW,
    eps,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b + res
    
    if do_relu:
        out = tl.maximum(out, tl.zeros_like(out))
    
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Implicit GEMM convolution kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    SH, SW, PH, PW,
    groups,
    M, N_out, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # M = N * OUT_H * OUT_W
    # N_out = OUT_C
    # K = IN_C_per_group * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    bn = offs_m // (OUT_W * OUT_H)
    
    # Determine group
    group_size_out = OUT_C // groups
    group_size_in = IN_C // groups
    g = offs_n // group_size_out
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    K_per_group = group_size_in * KH * KW
    
    for k in range(0, K_per_group, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c_local, kh, kw)
        in_c_local = offs_k % group_size_in
        kh = (offs_k // group_size_in) % KH
        kw = offs_k // (group_size_in * KH)
        
        # For each output channel group g, input channel = g * group_size_in + in_c_local
        # But g varies per output channel... handle per-group
        # Simplified: assume single group or handle groups=1
        # For groups > 1, we need per-element group computation
        
        ih = oh[:, None] * SH + kh[None, :] - PH
        iw = ow[:, None] * SW + kw[None, :] - PW
        
        # For grouped conv, input channel depends on output channel group
        # This is complex with variable groups - use g from offs_n
        # Since g varies per output channel, we need to handle this carefully
        # For simplicity, assume groups=1 path or depthwise
        in_c = in_c_local  # for groups=1
        
        valid_x = (bn[:, None] < N) & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (offs_k[None, :] < K_per_group)
        
        x_idx = bn[:, None] * IN_C * IN_H * IN_W + in_c[None, :] * IN_H * IN_W + ih * IN_W + iw
        x_tile = tl.load(x_ptr + x_idx, mask=valid_x, other=0.0)
        
        # Weight: [OUT_C, IN_C_per_group, KH, KW]
        # w_idx for output channel offs_n, in_c_local, kh, kw
        valid_w = (offs_n[:, None] < OUT_C) & (offs_k[None, :] < K_per_group)
        w_idx = offs_n[:, None] * K_per_group + offs_k[None, :]
        w_tile = tl.load(w_ptr + w_idx, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, tl.trans(w_tile), acc)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    out_idx = bn[:, None] * OUT_C * OUT_H * OUT_W + offs_n[None, :] * OUT_H * OUT_W + oh[:, None] * OUT_W + ow[:, None]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def triton_bn_relu(x, weight, bias, mean, var, eps=1e-5, relu=True):
    """Apply batch norm + optional relu using Triton."""
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        weight, bias, mean, var,
        numel, C, HW,
        eps,
        relu,
    )
    return out


def triton_channel_shuffle(x, groups):
    """Channel shuffle using Triton."""
    N, C, H, W = x.shape
    numel = N * C * H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid](
        x, out,
        N, C, H, W,
        groups,
        numel,
    )
    return out


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    """Max pool 2d using Triton."""
    N, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H + 2 * padding - kernel_size) // stride + 1
    OUT_W = (IN_W + 2 * padding - kernel_size) // stride + 1
    numel = N * C * OUT_H * OUT_W
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        kernel_size, kernel_size, stride, stride, padding, padding,
        numel,
    )
    return out


def triton_adaptive_avg_pool(x):
    """Adaptive avg pool to (1,1) using Triton."""
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(NC, meta['BLOCK_SIZE']),)
    adaptive_avg_pool_kernel[grid](
        x.reshape(N * C, H * W), out.reshape(N * C),
        NC, HW,
    )
    return out


def triton_linear(x, weight, bias):
    """Linear layer using Triton."""
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
    )
    return out


def triton_conv2d_groups1(x, weight, stride=1, padding=0):
    """Conv2d with groups=1 using implicit GEMM."""
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, IN_C_w, KH, KW = weight.shape
    assert IN_C == IN_C_w
    
    OUT_H = (IN_H + 2 * padding - KH) // stride + 1
    OUT_W = (IN_W + 2 * padding - KW) // stride + 1
    
    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW
    
    # Reshape weight to [OUT_C, K]
    w_flat = weight.reshape(OUT_C, K).contiguous()
    x_cont = x.contiguous()
    
    out = torch.empty(N, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    conv2d_implicit_gemm_kernel[grid](
        x_cont, w_flat, out,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride, stride, padding, padding,
        M, OUT_C, K,
    )
    return out


# ============================================================
# Grouped conv2d for ShuffleNet (groups > 1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N_per_group', 'K_per_group', 'groups'],
)
@triton.jit
def conv2d_grouped_kernel(
    x_ptr, w_ptr, out_ptr,
    N_batch, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    SH, SW, PH, PW,
    groups,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_ng = tl.program_id(1)  # combined (group, n_in_group) index
    
    # pid_ng encodes: group * cdiv(group_out_c, BLOCK_N) + n_block_in_group
    n_blocks_per_group = (N_per_group + BLOCK_N - 1) // BLOCK_N
    g = pid_ng // n_blocks_per_group
    pid_n = pid_ng % n_blocks_per_group
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_local = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = g * group_out_c + offs_n_local  # global output channel
    
    # Decode M -> (batch, out_h, out_w)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    bn = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K_per_group, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c_local, kh, kw)
        in_c_local = offs_k % group_in_c
        kh_idx = (offs_k // group_in_c) % KH
        kw_idx = offs_k // (group_in_c * KH)
        
        in_c_global = g * group_in_c + in_c_local
        
        ih = oh[:, None] * SH + kh_idx[None, :] - PH
        iw = ow[:, None] * SW + kw_idx[None, :] - PW
        
        valid_x = (bn[:, None] < N_batch) & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (offs_k[None, :] < K_per_group)
        
        x_idx = bn[:, None] * IN_C * IN_H * IN_W + in_c_global[None, :] * IN_H * IN_W + ih * IN_W + iw
        x_tile = tl.load(x_ptr + x_idx, mask=valid_x, other=0.0)
        
        # Weight layout: [OUT_C, group_in_c, KH, KW] -> [OUT_C, K_per_group]
        valid_w = (offs_n_local[:, None] < group_out_c) & (offs_k[None, :] < K_per_group)
        w_idx = offs_n[:, None] * K_per_group + offs_k[None, :]
        w_tile = tl.load(w_ptr + w_idx, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, tl.trans(w_tile), acc)
    
    out_mask = (offs_m[:, None] < M