import torch
import triton
import triton.language as tl
import math

# ─── Fused BN + optional activation kernels ───────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    ACT: tl.constexpr,   # 0=none, 1=relu, 2=hardtanh6
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # channel index: layout is (N, C, H, W) contiguous -> channel = (offset // HW) % C
    c_idx = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var  = tl.load(var_ptr  + c_idx, mask=mask, other=1.0)
    w    = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b    = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    y = x_norm * w + b

    if ACT == 1:
        y = tl.maximum(y, 0.0)
    elif ACT == 2:
        y = tl.minimum(tl.maximum(y, 0.0), 6.0)

    tl.store(out_ptr + offsets, y, mask=mask)


def bn_act(x, mean, var, weight, bias, act=0, eps=1e-5):
    """Fused BatchNorm + activation (0=none, 1=relu, 2=hardtanh6)."""
    out = torch.empty_like(x)
    numel = x.numel()
    C = x.shape[1]
    HW = x.shape[2] * x.shape[3] if x.ndim == 4 else 1

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, C, HW,
        eps,
        ACT=act,
    )
    return out


# ─── Fused BN + add + optional activation ─────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_add_kernel(
    x_ptr, residual_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c_idx = (offsets // HW) % C

    x   = tl.load(x_ptr       + offsets, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr   + c_idx, mask=mask, other=0.0)
    var  = tl.load(var_ptr    + c_idx, mask=mask, other=1.0)
    w    = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b    = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    y = x_norm * w + b + res

    tl.store(out_ptr + offsets, y, mask=mask)


def bn_add(x, residual, mean, var, weight, bias, eps=1e-5):
    """Fused BatchNorm + residual add."""
    out = torch.empty_like(x)
    numel = x.numel()
    C = x.shape[1]
    HW = x.shape[2] * x.shape[3] if x.ndim == 4 else 1

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_add_kernel[grid](
        x, residual, out,
        mean, var, weight, bias,
        numel, C, HW,
        eps,
    )
    return out


# ─── Fused GEMM (linear) kernel ───────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
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

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # add bias
    b_offs = offs_n
    b_mask = offs_n < N
    bias = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)
    acc = acc + bias[None, :]

    out_offs_m = offs_m[:, None]
    out_offs_n = offs_n[None, :]
    out_mask = (out_offs_m < M) & (out_offs_n < N)
    tl.store(out_ptr + out_offs_m * N + out_offs_n, acc, mask=out_mask)


def triton_linear(x, weight, bias):
    """x: (M, K), weight: (N, K), bias: (N,) -> (M, N)"""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


# ─── Adaptive avg pool + flatten ──────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def avg_pool_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    """Global average pool: output shape (N, C)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C

    n_idx = offsets // C
    c_idx = offsets % C

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for hw in range(HW):
        x_off = n_idx * C * HW + c_idx * HW + hw
        val = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        acc = acc + val

    acc = acc / HW
    tl.store(out_ptr + offsets, acc, mask=mask)


def triton_avg_pool(x):
    """x: (N, C, H, W) -> (N, C)"""
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty((N, C), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N * C, meta['BLOCK_SIZE']),)
    avg_pool_kernel[grid](x, out, N, C, HW)
    return out


# ─── Main forward function ────────────────────────────────────────────────────

def kernel_function(
    p_conv1_weight,
    p_bn1_weight, p_bn1_bias,
    p_blocks_0_depthwise_conv_0_weight,
    p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
    p_blocks_0_project_conv_0_weight,
    p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
    p_blocks_1_expand_conv_0_weight,
    p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
    p_blocks_1_depthwise_conv_0_weight,
    p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
    p_blocks_1_project_conv_0_weight,
    p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
    p_blocks_2_expand_conv_0_weight,
    p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
    p_blocks_2_depthwise_conv_0_weight,
    p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
    p_blocks_2_project_conv_0_weight,
    p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
    p_blocks_3_expand_conv_0_weight,
    p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
    p_blocks_3_depthwise_conv_0_weight,
    p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias,
    p_blocks_3_project_conv_0_weight,
    p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias,
    p_blocks_4_expand_conv_0_weight,
    p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias,
    p_blocks_4_depthwise_conv_0_weight,
    p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias,
    p_blocks_4_project_conv_0_weight,
    p_blocks_4_project_conv_1_weight, p_blocks_4_project_conv_1_bias,
    p_blocks_5_expand_conv_0_weight,
    p_blocks_5_expand_conv_1_weight, p_blocks_5_expand_conv_1_bias,
    p_blocks_5_depthwise_conv_0_weight,
    p_blocks_5_depthwise_conv_1_weight, p_blocks_5_depthwise_conv_1_bias,
    p_blocks_5_project_conv_0_weight,
    p_blocks_5_project_conv_1_weight, p_blocks_5_project_conv_1_bias,
    p_blocks_6_expand_conv_0_weight,
    p_blocks_6_expand_conv_1_weight, p_blocks_6_expand_conv_1_bias,
    p_blocks_6_depthwise_conv_0_weight,
    p_blocks_6_depthwise_conv_1_weight, p_blocks_6_depthwise_conv_1_bias,
    p_blocks_6_project_conv_0_weight,
    p_blocks_6_project_conv_1_weight, p_blocks_6_project_conv_1_bias,
    p_blocks_7_expand_conv_0_weight,
    p_blocks_7_expand_conv_1_weight, p_blocks_7_expand_conv_1_bias,
    p_blocks_7_depthwise_conv_0_weight,
    p_blocks_7_depthwise_conv_1_weight, p_blocks_7_depthwise_conv_1_bias,
    p_blocks_7_project_conv_0_weight,
    p_blocks_7_project_conv_1_weight, p_blocks_7_project_conv_1_bias,
    p_blocks_8_expand_conv_0_weight,
    p_blocks_8_expand_conv_1_weight, p_blocks_8_expand_conv_1_bias,
    p_blocks_8_depthwise_conv_0_weight,
    p_blocks_8_depthwise_conv_1_weight, p_blocks_8_depthwise_conv_1_bias,
    p_blocks_8_project_conv_0_weight,
    p_blocks_8_project_conv_1_weight, p_blocks_8_project_conv_1_bias,
    p_blocks_9_expand_conv_0_weight,
    p_blocks_9_expand_conv_1_weight, p_blocks_9_expand_conv_1_bias,
    p_blocks_9_depthwise_conv_0_weight,
    p_blocks_9_depthwise_conv_1_weight, p_blocks_9_depthwise_conv_1_bias,
    p_blocks_9_project_conv_0_weight,
    p_blocks_9_project_conv_1_weight, p_blocks_9_project_conv_1_bias,
    p_blocks_10_expand_conv_0_weight,
    p_blocks_10_expand_conv_1_weight, p_blocks_10_expand_conv_1_bias,
    p_blocks_10_depthwise_conv_0_weight,
    p_blocks_10_depthwise_conv_1_weight, p_blocks_10_depthwise_conv_1_bias,
    p_blocks_10_project_conv_0_weight,
    p_blocks_10_project_conv_1_weight, p_blocks_10_project_conv_1_bias,
    p_blocks_11_expand_conv_0_weight,
    p_blocks_11_expand_conv_1_weight, p_blocks_11_expand_conv_1_bias,
    p_blocks_11_depthwise_conv_0_weight,
    p_blocks_11_depthwise_conv_1_weight, p_blocks_11_depthwise_conv_1_bias,
    p_blocks_11_project_conv_0_weight,
    p_blocks_11_project_conv_1_weight, p_blocks_11_project_conv_1_bias,
    p_blocks_12_expand_conv_0_weight,
    p_blocks_12_expand_conv_1_weight, p_blocks_12_expand_conv_1_bias,
    p_blocks_12_depthwise_conv_0_weight,
    p_blocks_12_depthwise_conv_1_weight, p_blocks_12_depthwise_conv_1_bias,
    p_blocks_12_project_conv_0_weight,
    p_blocks_12_project_conv_1_weight, p_blocks_12_project_conv_1_bias,
    p_conv2_weight,
    p_bn2_weight, p_bn2_bias,
    p_fc_weight, p_fc_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
    b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
    b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
    b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
    b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
    b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
    b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
    b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,
    b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
    b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var,
    b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var,
    b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var,
    b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var,
    b_blocks_4_project_conv_1_running_mean, b_blocks_4_project_conv_1_running_var,
    b_blocks_5_expand_conv_1_running_mean, b_blocks_5_expand_conv_1_running_var,
    b_blocks_5_depthwise_conv_1_running_mean, b_blocks_5_depthwise_conv_1_running_var,
    b_blocks_5_project_conv_1_running_mean, b_blocks_5_project_conv_1_running_var,
    b_blocks_6_expand_conv_1_running_mean, b_blocks_6_expand_conv_1_running_var,
    b_blocks_6_depthwise_conv_1_running_mean, b_blocks_6_depthwise_conv_1_running_var,
    b_blocks_6_project_conv_1_running_mean, b_blocks_6_project_conv_1_running_var,
    b_blocks_7_expand_conv_1_running_mean, b_blocks_7_expand_conv_1_running_var,
    b_blocks_7_depthwise_conv_1_running_mean, b_blocks_7_depthwise_conv_1_running_var,
    b_blocks_7_project_conv_1_running_mean, b_blocks_7_project_conv_1_running_var,
    b_blocks_8_expand_conv_1_running_mean, b_blocks_8_expand_conv_1_running_var,
    b_blocks_8_depthwise_conv_1_running_mean, b_blocks_8_depthwise_conv_1_running_var,
    b_blocks_8_project_conv_1_running_mean, b_blocks_8_project_conv_1_running_var,
    b_blocks_9_expand_conv_1_running_mean, b_blocks_9_expand_conv_1_running_var,
    b_blocks_9_depthwise_conv_1_running_mean, b_blocks_9_depthwise_conv_1_running_var,
    b_blocks_9_project_conv_1_running_mean, b_blocks_9_project_conv_1_running_var,
    b_blocks_10_expand_conv_1_running_mean, b_blocks_10_expand_conv_1_running_var,
    b_blocks_10_depthwise_conv_1_running_mean, b_blocks_10_depthwise_conv_1_running_var,
    b_blocks_10_project_conv_1_running_mean, b_blocks_10_project_conv_1_running_var,
    b_blocks_11_expand_conv_1_running_mean, b_blocks_11_expand_conv_1_running_var,
    b_blocks_11_depthwise_conv_1_running_mean, b_blocks_11_depthwise_conv_1_running_var,
    b_blocks_11_project_conv_1_running_mean, b_blocks_11_project_conv_1_running_var,
    b_blocks_12_expand_conv_1_running_mean, b_blocks_12_expand_conv_1_running_var,
    b_blocks_12_depthwise_conv_1_running_mean, b_blocks_12_depthwise_conv_1_running_var,
    b_blocks_12_project_conv_1_running_mean, b_blocks_12_project_conv_1_running_var,
    b_bn2_running_mean, b_bn2_running_var,
    x,
):
    import torch.nn.functional as F

    # Helper: conv2d wrapper
    def conv2d(inp, weight, stride=1, padding=0, groups=1):
        return F.conv2d(inp, weight, bias=None, stride=stride, padding=padding, groups=groups)

    # ── Stem ──────────────────────────────────────────────────────────────────
    x = conv2d(x, p_conv1_weight, stride=2, padding=1)
    x = bn_act(x, b_bn1_running_mean, b_bn1_running_var,
               p_bn1_weight, p_bn1_bias, act=1)

    # ── Block 0: MBConv1 (32->16, k=3, s=1, expand=1) ─────────────────────────
    # depthwise (no expand since ratio=1)
    x0 = conv2d(x, p_blocks_0_depthwise_conv_0_weight, stride=1, padding=1, groups=32)
    x0 = bn_act(x0, b_blocks_0_depthwise_conv_1_running_mean,
                b_blocks_0_depthwise_conv_1_running_var,
                p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias, act=2)
    x0 = conv2d(x0, p_blocks_0_project_conv_0_weight, stride=1, padding=0)
    x0 = bn_act(x0, b_blocks_0_project_conv_1_running_mean,
                b_blocks_0_project_conv_1_running_var,
                p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias, act=0)
    # No residual for block 0 (in_channels != out_channels)
    x = x0

    # ── Block 1: MBConv6 (16->24, k=3, s=2, expand=6) ─────────────────────────
    x1 = conv2d(x, p_blocks_1_expand_conv_0_weight, stride=1, padding=0)
    x1 = bn_act(x1, b_blocks_1_expand_conv_1_running_mean,
                b_blocks_1_expand_conv_1_running_var,
                p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias, act=2)
    x1 = conv2d(x1, p_blocks_1_depthwise_conv_0_weight, stride=2, padding=1, groups=96)
    x1 = bn_act(x1, b_blocks_1_depthwise_conv_1_running_mean,
                b_blocks_1_depthwise_conv_1_running_var,
                p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias, act=2)
    x1 = conv2d(x1, p_blocks_1_project_conv_0_weight, stride=1, padding=0)
    x1 = bn_act(x1, b_blocks_1_project_conv_1_running_mean,
                b_blocks_1_project_conv_1_running_var,
                p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias, act=0)
    # No residual (stride=2, channels change)
    x = x1

    # ── Block 2: MBConv6 (24->24, k=3, s=1, expand=6) ─────────────────────────
    residual2 = x
    x2 = conv2d(x, p_blocks_2_expand_conv_0_weight, stride=1, padding=0)
    x2 = bn_act(x2, b_blocks_2_expand_conv_1_running_mean,
                b_blocks_2_expand_conv_1_running_var,
                p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias, act=2)
    x2 = conv2d(x2, p_blocks_2_depthwise_conv_0_weight, stride=1, padding=1, groups=144)
    x2 = bn_act(x2, b_blocks_2_depthwise_conv_1_running_mean,
                b_blocks_2_depthwise_conv_1_running_var,
                p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias, act=2)
    x2 = conv2d(x2, p_blocks_2_project_conv_0_weight, stride=1, padding=0)
    x2 = bn_add(x2, residual2,
                b_blocks_2_project_conv_1_running_mean,
                b_blocks_2_project_conv_1_running_var,
    x = x2

    # ── Block 3: MBConv6 (24->40, k=5, s=2, expand=6) ─────────────────────────
    x3 = conv2d(x, p_blocks_3_expand_conv_0_weight, stride=1, padding=0)
    x3 = bn_act(x3, b_blocks_3_expand_conv_1_running_mean,
                b_blocks_3_expand_conv_1_running_var,
                p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias, act=2)
    x3 = conv2d(x3, p_blocks_3_depthwise_conv_0_weight, stride=2, padding=2, groups=144)
    x3 = bn_act(x3, b_blocks_3_depthwise_conv_1_running_mean,