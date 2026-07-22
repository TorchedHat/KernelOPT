import torch
import triton
import triton.language as tl
import torch.nn.functional as F


import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # offsets layout: N*C*H*W, channel = (offset // HW) % C
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def apply_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5, relu=True):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = N * C * H * W
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, HW,
        eps=1e-5,
        do_relu=relu,
    )
    return out


# Channel shuffle kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    N, C, H, W, groups,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # output index: n, c_out, h, w
    # c_out = (c_in % groups) * (C // groups) + c_in // groups
    HW = H * W
    CHW = C * HW
    
    n = offsets // CHW
    rem = offsets % CHW
    c_out = rem // HW
    hw = rem % HW
    
    # inverse: c_out = g * cpg + r where g = c_in % groups, r = c_in // groups
    # c_in = r * groups + g
    cpg = C // groups
    g = c_out // cpg
    r = c_out % cpg
    c_in = r * groups + g
    
    src = n * CHW + c_in * HW + hw
    
    val = tl.load(x_ptr + src, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def channel_shuffle(x, groups):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = N * C * H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid](
        x, out,
        N, C, H, W, groups,
        numel,
    )
    return out


# Max pool 2d kernel (3x3, stride=2, padding=1)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
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
    COHW = C * OHW
    
    n = offsets // COHW
    rem = offsets % COHW
    c = rem // OHW
    rem2 = rem % OHW
    oh = rem2 // OUT_W
    ow = rem2 % OUT_W
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SW + kh - PH  # Note: using SW for SH here - fix below
            iw = ow * SW + kw - PW
            ih_real = oh * SH + kh - PH
            iw_real = ow * SW + kw - PW
            valid = mask & (ih_real >= 0) & (ih_real < IN_H) & (iw_real >= 0) & (iw_real < IN_W)
            src = n * C * IN_H * IN_W + c * IN_H * IN_W + ih_real * IN_W + iw_real
            val = tl.load(x_ptr + src, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


def max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    OUT_H = (H + 2 * padding - kernel_size) // stride + 1
    OUT_W = (W + 2 * padding - kernel_size) // stride + 1
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    numel = N * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out,
        N, C, H, W, OUT_H, OUT_W,
        kernel_size, kernel_size, stride, stride, padding, padding,
        numel,
    )
    return out


# Adaptive avg pool 2d (global) kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (n, c) pair
    pid = tl.program_id(0)
    nc_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    nc_mask = nc_offsets < N * C
    
    # For each (n, c), sum over HW
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for hw in range(HW):
        src = nc_offsets * HW + hw
        val = tl.load(x_ptr + src, mask=nc_mask, other=0.0).to(tl.float32)
        acc += val
    
    out = acc / HW
    tl.store(out_ptr + nc_offsets, out.to(tl.float32), mask=nc_mask)


def adaptive_avg_pool2d_global(x):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N * C, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x, out.view(N, C),
        N, C, HW,
    )
    return out


# Linear kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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
        
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=w_mask, other=0.0)
        
        acc = tl.dot(x, tl.trans(w), acc)
    
    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=out_mask)


def linear_triton(x, weight, bias):
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


# Conv2d using implicit GEMM in Triton
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    groups,
    M_TOTAL, TOTAL_K,
    s_xn, s_xc, s_xh, s_xw,
    s_wco, s_wci, s_wkh, s_wkw,
    s_yn, s_yc, s_yh, s_yw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)
    
    # Group handling
    GROUP_OUT_C = OUT_C // groups
    GROUP_IN_C = IN_C // groups
    GROUP_K = GROUP_IN_C * KH * KW
    
    group = offs_n // GROUP_OUT_C
    local_n = offs_n % GROUP_OUT_C
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_off in range(0, GROUP_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c_local, kh, kw)
        in_c_local = offs_k % GROUP_IN_C
        kw_idx = (offs_k // GROUP_IN_C) % KW
        kh_idx = offs_k // (GROUP_IN_C * KW)
        
        # Input coordinates
        ih = out_h[:, None] * SH + kh_idx[None, :] - PH
        iw = out_w[:, None] * SW + kw_idx[None, :] - PW
        
        # Global channel index
        in_c_global = group[:, None] * GROUP_IN_C + in_c_local[None, :]
        # Wait, group is per output channel, but for input we need per-batch group
        # Actually for grouped conv, each output channel group uses corresponding input group
        # group index for output n: group_n = offs_n // GROUP_OUT_C
        # But offs_m and offs_n are independent... need to handle per element
        # This is tricky with 2D tiling. Let's simplify for groups=1 first
        # For groups > 1, we need group per output channel
        
        valid = (
            (offs_m[:, None] < M_TOTAL) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < GROUP_K)
        )
        
        # For simplicity, handle group=1 case (most common for 1x1 convs)
        # in_c = in_c_local (when groups=1, GROUP_IN_C = IN_C)
        x_ptrs = (X_ptr + 
                  batch[:, None] * s_xn + 
                  in_c_local[None, :] * s_xc + 
        
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
        
        # Weight: [OUT_C, IN_C/groups, KH, KW]
        w_valid = (offs_n[:, None] < OUT_C) & (offs_k[None, :] < GROUP_K)
        w_ptrs = (W_ptr +
                  offs_n[:, None] * s_wco +
                  in_c_local[None, :] * s_wci +
                  kh_idx[None, :] * s_wkh +
                  kw_idx[None, :] * s_wkw)
        
        w_tile = tl.load(w_ptrs, mask=w_valid, other=0.0)
        
        acc = tl.dot(x_tile, tl.trans(w_tile), acc)
    
    # Store
    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr +
              batch[:, None] * s_yn +
              offs_n[None, :] * s_yc +
              out_h[:, None] * s_yh +
              out_w[:, None] * s_yw)
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def conv2d_triton(x, weight, stride=1, padding=0, groups=1):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, IN_C_per_group, KH, KW = weight.shape
    
    if isinstance(stride, int):
        SH = SW = stride
    else:
        SH, SW = stride
    if isinstance(padding, int):
        PH = PW = padding
    else:
        PH, PW = padding
    
    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1
    
    y = torch.empty(N, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    M_TOTAL = N * OUT_H * OUT_W
    GROUP_IN_C = IN_C // groups
    TOTAL_K = GROUP_IN_C * KH * KW
    
    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_kernel[grid](
        x, weight, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        groups,
        M_TOTAL, TOTAL_K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )
    return y


# Grouped conv2d for depthwise (groups=channels)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    OHW = OUT_H * OUT_W
    COHW = C * OHW
    
    n = offsets // COHW
    rem = offsets % COHW
    c = rem // OHW
    rem2 = rem % OHW
    oh = rem2 // OUT_W
    ow = rem2 % OUT_W
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            src = n * C * IN_H * IN_W + c * IN_H * IN_W + ih * IN_W + iw
            x_val = tl.load(X_ptr + src, mask=valid, other=0.0).to(tl.float32)
            w_val = tl.load(W_ptr + c * KH * KW + kh * KW + kw, mask=mask, other=0.0).to(tl.float32)
            acc += x_val * w_val
    
    tl.store(Y_ptr + offsets, acc.to(tl.float32), mask=mask)


def depthwise_conv2d_triton(x, weight, stride=1, padding=1):
    N, C, IN_H, IN_W = x.shape
    # weight shape: [C, 1, KH, KW]
    KH, KW = weight.shape[2], weight.shape[3]
    
    if isinstance(stride, int):
        SH = SW = stride
    else:
        SH, SW = stride
    if isinstance(padding, int):
        PH = PW = padding
    else:
        PH, PW = padding
    
    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1
    
    y = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    numel = N * C * OUT_H * OUT_W
    
    # Flatten weight to [C, KH*KW]
    w_flat = weight.view(C, KH * KW)
    # Need contiguous [C, KH, KW]
    w_cont = weight.view(C, KH, KW).contiguous()
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    depthwise_conv2d_kernel[grid](
        x, w_cont, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        numel,
    )
    return y


# Grouped conv2d (general groups, for group conv with groups=3)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M_TOTAL', 'OUT_C', 'GROUP_K'],
)
@triton.jit
def grouped_conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    groups, GROUP_IN_C, GROUP_OUT_C,
    M_TOTAL, GROUP_K,
    s_xn, s_xc, s_xh, s_xw,
    s_wco, s_wci, s_wkh, s_wkw,
    s_yn, s_yc, s_yh, s_yw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    out_h_idx = (offs_m // OUT_W) % OUT_H
    batch_idx = offs_m // (OUT_W * OUT_H)
    
    # Group for each output channel
    group_n = offs_n // GROUP_OUT_C
    local_n = offs_n % GROUP_OUT_C