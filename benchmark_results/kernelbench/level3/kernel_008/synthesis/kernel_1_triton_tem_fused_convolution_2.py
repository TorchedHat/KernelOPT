import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ─── Kernel 1: Implicit GEMM Conv2d (3x3, pad=1, stride=1) ───────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=4, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'IN_H', 'IN_W', 'OUT_H', 'OUT_W',
         'KERNEL_H', 'KERNEL_W', 'STRIDE_H', 'STRIDE_W', 'PAD_H', 'PAD_W'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, OUT_C: tl.constexpr,
    IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # strides for X (NCHW)
    s_xn: tl.constexpr, s_xc: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr,
    # strides for W (OIHW)
    s_wo: tl.constexpr, s_wi: tl.constexpr, s_wkh: tl.constexpr, s_wkw: tl.constexpr,
    # strides for Out (NCHW)
    s_on: tl.constexpr, s_oc: tl.constexpr, s_oh: tl.constexpr, s_ow: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M_TOTAL = BATCH * OUT_H * OUT_W

    # Decode M index
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K: k = ic * KH * KW + kh * KW + kw
        kw_idx = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh_idx = tmp_k % KERNEL_H
        ic_idx = tmp_k // KERNEL_H

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (
            (offs_m[:, None] < M_TOTAL) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * s_xn +
                  ic_idx[None, :] * s_xc +
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr +
                  offs_n[None, :] * s_wo +
                  ic_idx[:, None] * s_wi +
                  kh_idx[:, None] * s_wkh +
                  kw_idx[:, None] * s_wkw)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output NCHW
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    out_ptrs = (Out_ptr +
                batch_idx[:, None] * s_on +
                offs_n[None, :] * s_oc +
                out_h_idx[:, None] * s_oh +
                out_w_idx[:, None] * s_ow)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# ─── Kernel 2: Fused BN + ReLU (pointwise) ───────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N_TOTAL', 'N_CHANNELS'],
)
@triton.jit
def bn_relu_kernel(
    X_ptr, Out_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    N_TOTAL: tl.constexpr,
    N_CHANNELS: tl.constexpr,
    eps: tl.constexpr,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Channel index: layout is NCHW, channel = (flat_idx // (H*W)) % C
    # But we pass N_CHANNELS and spatial size implicitly via N_TOTAL
    # We need channel index from flat index
    # flat = n * C * H * W + c * H * W + h * W + w
    # channel = (flat // spatial_size) % C
    # We'll compute it as: channel = (offs // (N_TOTAL // (N_TOTAL // N_CHANNELS))) % N_CHANNELS
    # Actually: spatial_size = N_TOTAL // (BATCH * N_CHANNELS) -- but we don't have BATCH here
    # Better: pass spatial_size explicitly
    # For now, use: channel = (offs // spatial_per_channel) % N_CHANNELS
    # We'll pass spatial_per_channel as a constexpr
    # Actually let's just use offs % N_CHANNELS won't work for NCHW
    # For NCHW: channel = (offs // HW) % C where HW = N_TOTAL / (BATCH * C)
    # Let's pass HW as a separate param
    # We'll restructure: see bn_relu_kernel_v2 below
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    tl.store(Out_ptr + offs, x, mask=mask)


# Better BN+ReLU kernel with proper channel indexing
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N_TOTAL', 'SPATIAL'],
)
@triton.jit
def bn_relu_fused_kernel(
    X_ptr, Out_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    N_TOTAL,
    SPATIAL,
    eps,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # channel index for NCHW layout
    c_idx = (offs // SPATIAL) % (N_TOTAL // (SPATIAL * (N_TOTAL // (SPATIAL * 64))))
    # This is getting complicated. Let's just pass C directly.
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Out_ptr + offs, x, mask=mask)


# Clean BN+ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N_TOTAL', 'SPATIAL', 'C'],
)
@triton.jit
def bn_relu_kernel_v3(
    X_ptr, Out_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    N_TOTAL,
    SPATIAL,
    C,
    eps,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    c_idx = (offs // SPATIAL) % C

    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + c_idx, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + c_idx, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * gamma + beta

    if do_relu:
        out = tl.maximum(out, 0.0)

    tl.store(Out_ptr + offs, out, mask=mask)


# ─── Fused: BN + BN_downsample + Add + ReLU ──────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N_TOTAL', 'SPATIAL', 'C'],
)
@triton.jit
def bn_add_relu_kernel(
    X_ptr, Identity_ptr, Out_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    N_TOTAL,
    SPATIAL,
    C,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: BN(X) + Identity + ReLU"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    identity = tl.load(Identity_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    c_idx = (offs // SPATIAL) % C

    mean  = tl.load(mean_ptr  + c_idx, mask=mask, other=0.0).to(tl.float32)
    var   = tl.load(var_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + c_idx, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * gamma + beta + identity

    out = tl.maximum(out, 0.0)

    tl.store(Out_ptr + offs, out, mask=mask)


def launch_conv2d(x, w, out_h, out_w, pad_h, pad_w, stride_h, stride_w):
    """Launch implicit GEMM conv2d kernel."""
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = w.shape
    M = N * out_h * out_w

    out = torch.empty((N, C_out, out_h, out_w), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x, w, out,
        BATCH=N, IN_C=C_in, OUT_C=C_out,
        IN_H=H, IN_W=W,
        OUT_H=out_h, OUT_W=out_w,
        KERNEL_H=KH, KERNEL_W=KW,
        STRIDE_H=stride_h, STRIDE_W=stride_w,
        PAD_H=pad_h, PAD_W=pad_w,
        s_xn=x.stride(0), s_xc=x.stride(1), s_xh=x.stride(2), s_xw=x.stride(3),
        s_wo=w.stride(0), s_wi=w.stride(1), s_wkh=w.stride(2), s_wkw=w.stride(3),
        s_on=out.stride(0), s_oc=out.stride(1), s_oh=out.stride(2), s_ow=out.stride(3),
    )
    return out


def launch_bn_relu(x, mean, var, gamma, beta, eps=1e-5, do_relu=True):
    N, C, H, W = x.shape
    N_TOTAL = N * C * H * W
    SPATIAL = H * W
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    bn_relu_kernel_v3[grid](
        x, out,
        mean, var, gamma, beta,
        N_TOTAL, SPATIAL, C, eps,
        do_relu=do_relu,
    )
    return out


def launch_bn_add_relu(x, identity, mean, var, gamma, beta, eps=1e-5):
    N, C, H, W = x.shape
    N_TOTAL = N * C * H * W
    SPATIAL = H * W
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    bn_add_relu_kernel[grid](
        x, identity, out,
        mean, var, gamma, beta,
        N_TOTAL, SPATIAL, C, eps,
    )
    return out


def kernel_function(
    p_conv1_weight,
    p_bn1_weight,
    p_bn1_bias,
    p_conv2_weight,
    p_bn2_weight,
    p_bn2_bias,
    p_downsample_0_weight,
    p_downsample_1_weight,
    p_downsample_1_bias,
    b_bn1_running_mean,
    b_bn1_running_var,
    b_bn2_running_mean,
    b_bn2_running_var,
    b_downsample_1_running_mean,
    b_downsample_1_running_var,
    x,
):
    N, C_in, H, W = x.shape  # [10, 3, 224, 224]

    # ── Conv1: 3x3, pad=1, stride=1 ──────────────────────────────────────────
    out1 = launch_conv2d(x, p_conv1_weight,
                         out_h=H, out_w=W,
                         pad_h=1, pad_w=1,
                         stride_h=1, stride_w=1)

    # ── BN1 + ReLU ────────────────────────────────────────────────────────────
    out1 = launch_bn_relu(out1,
                          b_bn1_running_mean, b_bn1_running_var,
                          p_bn1_weight, p_bn1_bias,
                          eps=1e-5, do_relu=True)

    # ── Conv2: 3x3, pad=1, stride=1 ──────────────────────────────────────────
    out2 = launch_conv2d(out1, p_conv2_weight,
                         out_h=H, out_w=W,
                         pad_h=1, pad_w=1,
                         stride_h=1, stride_w=1)

    # ── Downsample Conv: 1x1, pad=0, stride=1 ────────────────────────────────
    identity = launch_conv2d(x, p_downsample_0_weight,
                             out_h=H, out_w=W,
                             pad_h=0, pad_w=0,
                             stride_h=1, stride_w=1)

    # ── BN_downsample ─────────────────────────────────────────────────────────
    identity = launch_bn_relu(identity,
                              b_downsample_1_running_mean, b_downsample_1_running_var,
                              p_downsample_1_weight, p_downsample_1_bias,
                              eps=1e-5, do_relu=False)

    # ── BN2 + Add + ReLU ──────────────────────────────────────────────────────
    out = launch_bn_add_relu(out2, identity,
                             b_bn2_running_mean, b_bn2_running_var,
                             p_bn2_weight, p_bn2_bias,
                             eps=1e-5)

    return out


def get_inputs():
    in_channels = 3
    out_channels = 64
    batch_size = 10
    H, W = 224, 224

    p_conv1_weight = torch.randn(out_channels, in_channels, 3, 3)
    p_bn1_weight = torch.ones(out_channels)
    p_bn1_bias = torch.zeros(out_channels)
    p_conv2_weight = torch.randn(out_channels, out_channels, 3, 3)
    p_bn2_weight = torch.ones(out_channels)
    p_bn2_bias = torch.zeros(out_channels)
    p_downsample_0_weight = torch.randn(out_channels, in_channels, 1, 1)
    p_downsample_1_weight = torch.ones(out_channels)
    p_downsample_1_bias = torch.zeros(out_channels)
    b_bn1_running_mean = torch.zeros(out_channels)
    b_bn1_running_var = torch.ones(out_channels)
    b_bn2_running_mean = torch.zeros(out_channels)
    b_bn2_running_var = torch.ones(out_channels)
    b_downsample_1_running_mean = torch.zeros(out_channels)
    b_downsample_1_running_var = torch.ones(out_channels)
    x = torch.randn(batch_size, in_channels, H, W)

    return [
        p_conv1_weight.cuda(),
        p_bn1_weight.cuda(),
        p_bn1_bias.cuda(),
        p_conv2_weight.cuda(),
        p_bn2_weight.cuda(),
        p_bn2_bias.cuda(),
        p_downsample_0_weight.cuda(),
        p_downsample_1_weight.cuda(),
        p_downsample_1_bias.cuda(),
        b_bn1_running_mean.cuda(),
        b_bn1_running_var.cuda(),
        b_bn2_running_mean.cuda(),
        b_bn2_running_var.cuda(),
        b_downsample_1_running_mean.cuda(),
        b_downsample_1_running_var.cuda(),
        x.cuda(),
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn

        (p_conv1_weight, p_bn1_weight, p_bn1_bias,
         p_conv2_weight, p_bn2_weight, p_bn2_bias,
         p_downsample_0_weight, p_downsample_1_weight, p_downsample_1_bias,
         b_bn1_running_mean, b_bn1_running_var,
         b_bn2_running_mean, b_bn2_running_var,
         b_downsample_1_running_mean, b_downsample_1_running_var,
         x) = inputs

        # Reference PyTorch computation
        conv1_out = torch.nn.functional.conv2d(x, p_conv1_weight, bias=None,
                                                stride=1, padding=1)
        bn1_out = torch.nn.functional.batch_norm(
            conv1_out, b_bn1_running_mean, b_bn1_running_var,
            p_bn1_weight, p_bn1_bias, training=False, eps=1e-5)
        relu1_out = torch.nn.functional.relu(bn1_out)

        conv2_out = torch.nn.functional.conv2d(relu1_out, p_conv2_weight, bias=None,
                                                stride=1, padding=1)
        bn2_out = torch.nn.functional.batch_norm(
            conv2_out, b_bn2_running_mean, b_bn2_running_var,
            p_bn2_weight, p_bn2_bias, training=False, eps=1e-5)

        ds_conv_out = torch.nn.functional.conv2d(x, p_downsample_0_weight, bias=None,
                                                  stride=1, padding=0)
        ds_bn_out = torch.nn.functional.batch_norm(
            ds_conv_out, b_downsample_1_running_mean, b_downsample_1_running_var,
            p_downsample_1_weight, p_downsample_1_bias, training=False, eps=1e-5)

        ref = torch.nn.functional.relu(bn2_out + ds_bn_out)

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")