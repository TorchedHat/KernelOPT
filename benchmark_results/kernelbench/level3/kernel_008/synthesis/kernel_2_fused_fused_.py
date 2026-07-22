import torch
import triton
import triton.language as tl


# ─── Fused BN + ReLU pointwise kernel ────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
    ],
    key=['numel'],
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

    # Determine channel index: layout is (N, C, H, W) -> channel = (offset // HW) % C
    c_idx = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + c_idx, mask=mask, other=1.0).to(tl.float32)
    w    = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    y = x_norm * w + b

    if do_relu:
        y = tl.maximum(y, 0.0)

    tl.store(out_ptr + offsets, y.to(tl.float32), mask=mask)


# ─── Fused BN + add + ReLU pointwise kernel ──────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def bn_add_relu_kernel(
    x_ptr, identity_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c_idx = (offsets // HW) % C

    x    = tl.load(x_ptr        + offsets, mask=mask, other=0.0).to(tl.float32)
    idn  = tl.load(identity_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr    + c_idx, mask=mask, other=1.0).to(tl.float32)
    w    = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    y = x_norm * w + b + idn
    y = tl.maximum(y, 0.0)

    tl.store(out_ptr + offsets, y.to(tl.float32), mask=mask)


# ─── Implicit GEMM Conv2d kernel ─────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    KERNEL_H, KERNEL_W,
    M_TOTAL, TOTAL_K,
    # strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # strides for W (out_c, kh, kw, in_c) layout
    s_wo, s_wkh, s_wkw, s_wi,
    # strides for Y (NCHW layout)
    s_yn, s_yc, s_yh, s_yw,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K -> (kh, kw, in_c)
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KERNEL_W
        kh_idx = tmp_k // KERNEL_W

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_spatial

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k[:, None] & valid_n
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output in NCHW layout
    valid_m_out = offs_m[:, None] < M_TOTAL
    valid_n_out = offs_n[None, :] < OUT_C

    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_m_out & valid_n_out)


def launch_conv2d(x, w, stride_h, stride_w, pad_h, pad_w):
    """
    returns: (N, out_c, out_h, out_w) float32
    """
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = w.shape
    OUT_H = (IN_H + 2 * pad_h - KERNEL_H) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KERNEL_W) // stride_w + 1

    # Convert input to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # Convert weight to (out_c, kh, kw, in_c)
    w_conv = w.permute(0, 2, 3, 1).contiguous()

    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = N * OUT_H * OUT_W
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_conv, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        stride_h, stride_w,
        pad_h, pad_w,
        KERNEL_H, KERNEL_W,
        M_TOTAL, TOTAL_K,
        # strides for x_nhwc: (N, H, W, C)
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        # strides for w_conv: (out_c, kh, kw, in_c)
        w_conv.stride(0), w_conv.stride(1), w_conv.stride(2), w_conv.stride(3),
        # strides for y: (N, C, H, W)
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )
    return y


def launch_bn_relu(x, mean, var, weight, bias, eps=1e-5, do_relu=True):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, C, HW,
        eps, do_relu,
    )
    return out


def launch_bn_add_relu(x, identity, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_add_relu_kernel[grid](
        x, identity, out,
        mean, var, weight, bias,
        numel, C, HW,
        eps,
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
    # conv1: 3x3, stride=1, pad=1
    out = launch_conv2d(x, p_conv1_weight, 1, 1, 1, 1)
    # bn1 + relu
    out = launch_bn_relu(out, b_bn1_running_mean, b_bn1_running_var,
                         p_bn1_weight, p_bn1_bias, eps=1e-5, do_relu=True)

    # conv2: 3x3, stride=1, pad=1
    out = launch_conv2d(out, p_conv2_weight, 1, 1, 1, 1)
    # bn2 (no relu yet -- will be fused with add+relu)

    # downsample: 1x1, stride=1, pad=0
    identity = launch_conv2d(x, p_downsample_0_weight, 1, 1, 0, 0)
    # downsample bn
    identity = launch_bn_relu(identity, b_downsample_1_running_mean,
                               b_downsample_1_running_var,
                               p_downsample_1_weight, p_downsample_1_bias,
                               eps=1e-5, do_relu=False)

    # bn2 + add identity + relu
    out = launch_bn_add_relu(out, identity,
                              b_bn2_running_mean, b_bn2_running_var,
                              p_bn2_weight, p_bn2_bias, eps=1e-5)
    return out


def get_inputs():
    torch.manual_seed(0)
    p_conv1_weight = torch.randn(64, 3, 3, 3)
    p_bn1_weight = torch.ones(64)
    p_bn1_bias = torch.zeros(64)
    p_conv2_weight = torch.randn(64, 64, 3, 3)
    p_bn2_weight = torch.ones(64)
    p_bn2_bias = torch.zeros(64)
    p_downsample_0_weight = torch.randn(64, 3, 1, 1)
    p_downsample_1_weight = torch.ones(64)
    p_downsample_1_bias = torch.zeros(64)
    b_bn1_running_mean = torch.zeros(64)
    b_bn1_running_var = torch.ones(64)
    b_bn2_running_mean = torch.zeros(64)
    b_bn2_running_var = torch.ones(64)
    b_downsample_1_running_mean = torch.zeros(64)
    b_downsample_1_running_var = torch.ones(64)
    x = torch.randn(10, 3, 224, 224)
    return [
        p_conv1_weight, p_bn1_weight, p_bn1_bias,
        p_conv2_weight, p_bn2_weight, p_bn2_bias,
        p_downsample_0_weight, p_downsample_1_weight, p_downsample_1_bias,
        b_bn1_running_mean, b_bn1_running_var,
        b_bn2_running_mean, b_bn2_running_var,
        b_downsample_1_running_mean, b_downsample_1_running_var,
        x,
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    inputs_cuda = [t.cuda() for t in inputs]

    output = kernel_function(*inputs_cuda)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn

        (p_conv1_weight, p_bn1_weight, p_bn1_bias,
         p_conv2_weight, p_bn2_weight, p_bn2_bias,
         p_downsample_0_weight, p_downsample_1_weight, p_downsample_1_bias,
         b_bn1_running_mean, b_bn1_running_var,
         b_bn2_running_mean, b_bn2_running_var,
         b_downsample_1_running_mean, b_downsample_1_running_var,
         x) = inputs_cuda

        # Reference using PyTorch eager
        conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False).cuda()
        conv1.weight.data.copy_(p_conv1_weight)
        bn1 = nn.BatchNorm2d(64).cuda()
        bn1.weight.data.copy_(p_bn1_weight)
        bn1.bias.data.copy_(p_bn1_bias)
        bn1.running_mean.copy_(b_bn1_running_mean)
        bn1.running_var.copy_(b_bn1_running_var)
        bn1.eval()

        conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False).cuda()
        conv2.weight.data.copy_(p_conv2_weight)
        bn2 = nn.BatchNorm2d(64).cuda()
        bn2.weight.data.copy_(p_bn2_weight)
        bn2.bias.data.copy_(p_bn2_bias)
        bn2.running_mean.copy_(b_bn2_running_mean)
        bn2.running_var.copy_(b_bn2_running_var)
        bn2.eval()

        ds_conv = nn.Conv2d(3, 64, 1, stride=1, bias=False).cuda()
        ds_conv.weight.data.copy_(p_downsample_0_weight)
        ds_bn = nn.BatchNorm2d(64).cuda()
        ds_bn.weight.data.copy_(p_downsample_1_weight)
        ds_bn.bias.data.copy_(p_downsample_1_bias)
        ds_bn.running_mean.copy_(b_downsample_1_running_mean)
        ds_bn.running_var.copy_(b_downsample_1_running_var)
        ds_bn.eval()

        with torch.no_grad():
            identity = x
            out_ref = conv1(x)
            out_ref = bn1(out_ref)
            out_ref = torch.relu(out_ref)
            out_ref = conv2(out_ref)
            out_ref = bn2(out_ref)
            id_ref = ds_conv(identity)
            id_ref = ds_bn(id_ref)
            out_ref = out_ref + id_ref
            out_ref = torch.relu(out_ref)

        max_diff = (output - out_ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, out_ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")