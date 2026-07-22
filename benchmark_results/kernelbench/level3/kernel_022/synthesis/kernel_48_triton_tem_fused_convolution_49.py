import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# ─── Fused BatchNorm + optional activation (relu / hardtanh / none) ───────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def fused_bn_act_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, spatial,
    eps,
    act,          # 0=none, 1=relu, 2=hardtanh
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # channel index: layout is (N, C, H, W) contiguous -> channel = (offset // spatial) % C
    c_idx = (offsets // spatial) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + c_idx, mask=mask, other=1.0).to(tl.float32)
    w    = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (x - mean) * inv_std * w + b

    # activation
    if act == 1:  # relu
        y = tl.where(y > 0.0, y, 0.0)
    elif act == 2:  # hardtanh [-6, 6]
        y = tl.where(y < -6.0, -6.0, y)
        y = tl.where(y > 6.0,  6.0,  y)

    tl.store(out_ptr + offsets, y.to(tl.float32), mask=mask)


def fused_bn_act(x, running_mean, running_var, weight, bias, eps=1e-5, act=0):
    """act: 0=none, 1=relu, 2=hardtanh"""
    out = torch.empty_like(x)
    numel = x.numel()
    C = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_act_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, spatial,
        eps, act,
    )
    return out


# ─── Fused BN + act in-place (for residual add path) ─────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def fused_bn_add_kernel(
    x_ptr, residual_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """BN(x) + residual, no activation (project conv in MBConv)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c_idx = (offsets // spatial) % C

    x   = tl.load(x_ptr       + offsets, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr    + c_idx, mask=mask, other=1.0).to(tl.float32)
    w    = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (x - mean) * inv_std * w + b + res

    tl.store(out_ptr + offsets, y.to(tl.float32), mask=mask)


def fused_bn_add(x, residual, running_mean, running_var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    C = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_add_kernel[grid](
        x, residual, out,
        running_mean, running_var, weight, bias,
        numel, C, spatial, eps,
    )
    return out


# ─── Fused linear (GEMM epilogue: y = x @ W^T + b) via Triton tl.dot ─────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=2),
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        mask_x = (offs_m[:, None] < M) & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        mask_w = (offs_n[None, :] < N) & mask_k[:, None]
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    offs_n2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b = tl.load(b_ptr + offs_n2, mask=offs_n2 < N, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n2[None, :]
    mask_out = (offs_m[:, None] < M) & (offs_n2[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


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


# ─── Adaptive avg pool 2d (global) via Triton ─────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['spatial'],
)
@triton.jit
def global_avg_pool_kernel(
    x_ptr, out_ptr,
    N, C, spatial,
    BLOCK_SIZE: tl.constexpr,
):
    """Global average pool: output shape (N, C, 1, 1)"""
    pid = tl.program_id(0)  # one program per (n, c) pair
    nc = pid
    n = nc // C
    c = nc % C

    base = n * C * spatial + c * spatial
    acc = tl.zeros((), dtype=tl.float32)
    count = 0

    for start in range(0, spatial, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < spatial
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    acc = acc / spatial
    tl.store(out_ptr + pid, acc.to(tl.float32))


def triton_global_avg_pool(x):
    N, C, H, W = x.shape
    spatial = H * W
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    global_avg_pool_kernel[grid](
        x, out,
        N, C, spatial,
    )
    return out


# ─── Conv2d helpers (using cuDNN) ─────────────────────────────────────────────
def conv2d(x, w, stride=1, padding=0, groups=1):
    return F.conv2d(x, w, bias=None, stride=stride, padding=padding, groups=groups)


# ─── Main kernel function ──────────────────────────────────────────────────────
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
    # ── Stem ──────────────────────────────────────────────────────────────────
    x = conv2d(x, p_conv1_weight, stride=2, padding=1)
    x = fused_bn_act(x, b_bn1_running_mean, b_bn1_running_var,
                     p_bn1_weight, p_bn1_bias, act=1)  # relu

    # ── Block 0: MBConv1(32->16, k=3, s=1) -- no expand, no residual ───────────
    # depthwise
    x0 = conv2d(x, p_blocks_0_depthwise_conv_0_weight, stride=1, padding=1,
                groups=32)
    x0 = fused_bn_act(x0, b_blocks_0_depthwise_conv_1_running_mean,
                      b_blocks_0_depthwise_conv_1_running_var,
                      p_blocks_0_depthwise_conv_1_weight,
                      p_blocks_0_depthwise_conv_1_bias, act=2)  # hardtanh
    # project
    x0 = conv2d(x0, p_blocks_0_project_conv_0_weight, stride=1, padding=0)
    x0 = fused_bn_act(x0, b_blocks_0_project_conv_1_running_mean,
                      b_blocks_0_project_conv_1_running_var,
                      p_blocks_0_project_conv_1_weight,
                      p_blocks_0_project_conv_1_bias, act=0)  # none
    x = x0  # no residual (in_c != out_c)

    # ── Block 1: MBConv6(16->24, k=3, s=2) -- no residual ─────────────────────
    x1 = conv2d(x, p_blocks_1_expand_conv_0_weight, stride=1, padding=0)
    x1 = fused_bn_act(x1, b_blocks_1_expand_conv_1_running_mean,
                      b_blocks_1_expand_conv_1_running_var,
                      p_blocks_1_expand_conv_1_weight,
                      p_blocks_1_expand_conv_1_bias, act=2)
    x1 = conv2d(x1, p_blocks_1_depthwise_conv_0_weight, stride=2, padding=1,
                groups=96)
    x1 = fused_bn_act(x1, b_blocks_1_depthwise_conv_1_running_mean,
                      b_blocks_1_depthwise_conv_1_running_var,
                      p_blocks_1_depthwise_conv_1_weight,
                      p_blocks_1_depthwise_conv_1_bias, act=2)
    x1 = conv2d(x1, p_blocks_1_project_conv_0_weight, stride=1, padding=0)
    x1 = fused_bn_act(x1, b_blocks_1_project_conv_1_running_mean,
                      b_blocks_1_project_conv_1_running_var,
                      p_blocks_1_project_conv_1_weight,
                      p_blocks_1_project_conv_1_bias, act=0)
    x = x1  # no residual (stride=2)

    # ── Block 2: MBConv6(24->24, k=3, s=1) -- residual ─────────────────────────
    x2 = conv2d(x, p_blocks_2_expand_conv_0_weight, stride=1, padding=0)
    x2 = fused_bn_act(x2, b_blocks_2_expand_conv_1_running_mean,
                      b_blocks_2_expand_conv_1_running_var,
                      p_blocks_2_expand_conv_1_weight,
                      p_blocks_2_expand_conv_1_bias, act=2)
    x2 = conv2d(x2, p_blocks_2_depthwise_conv_0_weight, stride=1, padding=1,
                groups=144)
    x2 = fused_bn_act(x2, b_blocks_2_depthwise_conv_1_running_mean,
                      b_blocks_2_depthwise_conv_1_running_var,
                      p_blocks_2_depthwise_conv_1_weight,
                      p_blocks_2_depthwise_conv_1_bias, act=2)
    x2 = conv2d(x2, p_blocks_2_project_conv_0_weight, stride=1, padding=0)
    x2 = fused_bn_add(x2, x,
                      b_blocks_2_project_conv_1_running_mean,