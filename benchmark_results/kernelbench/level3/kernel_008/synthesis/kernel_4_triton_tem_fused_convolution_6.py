import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Implicit-GEMM Conv2d (general)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=4, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W',
         'KERNEL_H', 'KERNEL_W', 'STRIDE_H', 'STRIDE_W', 'PAD_H', 'PAD_W'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # strides for X (NHWC layout)
    s_xn: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr, s_xc: tl.constexpr,
    # strides for W (OUT_C, IN_C, KH, KW) -> stored as (OUT_C, KH, KW, IN_C) after permute
    s_wo: tl.constexpr, s_wkh: tl.constexpr, s_wkw: tl.constexpr, s_wi: tl.constexpr,
    # strides for Y (NHWC layout)
    s_yn: tl.constexpr, s_yh: tl.constexpr, s_yw: tl.constexpr, s_yc: tl.constexpr,
    M_TOTAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch  = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)  [stored as OHWI weight]
        in_c = offs_k % IN_C
        kw   = (offs_k // IN_C) % KERNEL_W
        kh   = offs_k // (IN_C * KERNEL_W)

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = (batch < BATCH) & (offs_m < M_TOTAL)
        valid_k = offs_k < TOTAL_K
        valid = valid_m[:, None] & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & valid_k[None, :]

        x_ptrs = X_ptr + (batch[:, None] * s_xn + ih * s_xh + iw * s_xw + in_c[None, :] * s_xc)
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # W stored as (OUT_C, KH, KW, IN_C)
        w_ptrs = W_ptr + (offs_n[None, :] * s_wo + kh[:, None] * s_wkh + kw[:, None] * s_wkw + in_c[:, None] * s_wi)
        mask_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Store Y in NHWC
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + (batch[:, None] * s_yn + out_h[:, None] * s_yh + out_w[:, None] * s_yw + offs_n[None, :] * s_yc)
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Fused BN + ReLU (pointwise, NHWC input)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=2, num_warps=8),
    ],
    key=['N_TOTAL', 'C'],
)
@triton.jit
def bn_relu_kernel(
    X_ptr, Y_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    N_TOTAL, C: tl.constexpr,
    eps: tl.constexpr,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # channel index: in NHWC layout, channel is the last dim
    c_idx = offs % C

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + c_idx, mask=mask, other=1.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + c_idx, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * gamma + beta

    if do_relu:
        out = tl.maximum(out, tl.zeros_like(out))

    tl.store(Y_ptr + offs, out.to(tl.float32), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Fused BN + Add + ReLU (pointwise, NHWC input)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=2, num_warps=8),
    ],
    key=['N_TOTAL', 'C'],
)
@triton.jit
def bn_add_relu_kernel(
    X_ptr, Identity_ptr, Y_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    N_TOTAL, C: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    c_idx = offs % C

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    identity = tl.load(Identity_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean  = tl.load(mean_ptr  + c_idx, mask=mask, other=0.0).to(tl.float32)
    var   = tl.load(var_ptr   + c_idx, mask=mask, other=1.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + c_idx, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * gamma + beta + identity

    out = tl.maximum(out, tl.zeros_like(out))

    tl.store(Y_ptr + offs, out.to(tl.float32), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: launch conv2d implicit GEMM
# ─────────────────────────────────────────────────────────────────────────────
def _conv2d(x_nhwc, w_ohwi, out_h, out_w, stride_h, stride_w, pad_h, pad_w):
    """
    returns: y_nhwc (N, out_h, out_w, C_out)
    """
    N, H, W, C_in = x_nhwc.shape
    C_out, KH, KW, _ = w_ohwi.shape

    M_TOTAL = N * out_h * out_w
    y_nhwc = torch.empty((N, out_h, out_w, C_out), device=x_nhwc.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(C_out,   meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_ohwi, y_nhwc,
        BATCH=N, IN_C=C_in, IN_H=H, IN_W=W,
        OUT_C=C_out, OUT_H=out_h, OUT_W=out_w,
        KERNEL_H=KH, KERNEL_W=KW,
        STRIDE_H=stride_h, STRIDE_W=stride_w,
        PAD_H=pad_h, PAD_W=pad_w,
        s_xn=x_nhwc.stride(0), s_xh=x_nhwc.stride(1),
        s_xw=x_nhwc.stride(2), s_xc=x_nhwc.stride(3),
        s_wo=w_ohwi.stride(0), s_wkh=w_ohwi.stride(1),
        s_wkw=w_ohwi.stride(2), s_wi=w_ohwi.stride(3),
        s_yn=y_nhwc.stride(0), s_yh=y_nhwc.stride(1),
        s_yw=y_nhwc.stride(2), s_yc=y_nhwc.stride(3),
        M_TOTAL=M_TOTAL,
    )
    return y_nhwc


def _bn_relu(x_nhwc, mean, var, gamma, beta, eps=1e-5, do_relu=True):
    N_TOTAL = x_nhwc.numel()
    C = x_nhwc.shape[-1]
    y = torch.empty_like(x_nhwc)
    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x_nhwc, y,
        mean, var, gamma, beta,
        N_TOTAL=N_TOTAL, C=C,
        eps=eps, do_relu=do_relu,
    )
    return y


def _bn_add_relu(x_nhwc, identity_nhwc, mean, var, gamma, beta, eps=1e-5):
    N_TOTAL = x_nhwc.numel()
    C = x_nhwc.shape[-1]
    y = torch.empty_like(x_nhwc)
    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    bn_add_relu_kernel[grid](
        x_nhwc, identity_nhwc, y,
        mean, var, gamma, beta,
        N_TOTAL=N_TOTAL, C=C,
        eps=eps,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel function
# ─────────────────────────────────────────────────────────────────────────────
def kernel_function(
):
    N, C_in, H, W = x.shape  # 10, 3, 224, 224

    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (10, 224, 224, 3)

    # ── conv1: 3x3, stride=1, pad=1, in_c=3, out_c=64 ──
    # Weight: (64, 3, 3, 3) -> permute to (64, 3, 3, 3) OHWI
    w1 = p_conv1_weight.permute(0, 2, 3, 1).contiguous()  # (64, 3, 3, 3)
    out_h1 = (H + 2*1 - 3) // 1 + 1  # 224
    out_w1 = (W + 2*1 - 3) // 1 + 1  # 224
    conv1_out = _conv2d(x_nhwc, w1, out_h1, out_w1, 1, 1, 1, 1)  # (10, 224, 224, 64)

    # ── bn1 + relu ──
    bn1_out = _bn_relu(conv1_out, b_bn1_running_mean, b_bn1_running_var,
                       p_bn1_weight, p_bn1_bias, eps=1e-5, do_relu=True)  # (10, 224, 224, 64)

    # ── conv2: 3x3, stride=1, pad=1, in_c=64, out_c=64 ──
    w2 = p_conv2_weight.permute(0, 2, 3, 1).contiguous()  # (64, 3, 3, 64)
    out_h2 = (out_h1 + 2*1 - 3) // 1 + 1  # 224
    out_w2 = (out_w1 + 2*1 - 3) // 1 + 1  # 224
    conv2_out = _conv2d(bn1_out, w2, out_h2, out_w2, 1, 1, 1, 1)  # (10, 224, 224, 64)

    # ── downsample conv: 1x1, stride=1, pad=0, in_c=3, out_c=64 ──
    w_ds = p_downsample_0_weight.permute(0, 2, 3, 1).contiguous()  # (64, 1, 1, 3)
    out_h_ds = (H + 2*0 - 1) // 1 + 1  # 224
    out_w_ds = (W + 2*0 - 1) // 1 + 1  # 224
    conv_ds_out = _conv2d(x_nhwc, w_ds, out_h_ds, out_w_ds, 1, 1, 0, 0)  # (10, 224, 224, 64)

    # ── downsample bn (no relu) ──
    bn_ds_out = _bn_relu(conv_ds_out, b_downsample_1_running_mean, b_downsample_1_running_var,
                         p_downsample_1_weight, p_downsample_1_bias, eps=1e-5, do_relu=False)

    # ── bn2 + add(identity) + relu ──
    final_out = _bn_add_relu(conv2_out, bn_ds_out,
                             b_bn2_running_mean, b_bn2_running_var,
                             p_bn2_weight, p_bn2_bias, eps=1e-5)  # (10, 224, 224, 64)

    # Convert back to NCHW
    result = final_out.permute(0, 3, 1, 2).contiguous()  # (10, 64, 224, 224)
    return result


def get_inputs():
    torch.manual_seed(42)
    p_conv1_weight        = torch.randn(64, 3, 3, 3,  dtype=torch.float32, device='cuda')
    p_bn1_weight          = torch.ones(64,            dtype=torch.float32, device='cuda')
    p_bn1_bias            = torch.zeros(64,           dtype=torch.float32, device='cuda')
    p_conv2_weight        = torch.randn(64, 64, 3, 3, dtype=torch.float32, device='cuda')
    p_bn2_weight          = torch.ones(64,            dtype=torch.float32, device='cuda')
    p_bn2_bias            = torch.zeros(64,           dtype=torch.float32, device='cuda')
    p_downsample_0_weight = torch.randn(64, 3, 1, 1,  dtype=torch.float32, device='cuda')
    p_downsample_1_weight = torch.ones(64,            dtype=torch.float32, device='cuda')
    p_downsample_1_bias   = torch.zeros(64,           dtype=torch.float32, device='cuda')
    b_bn1_running_mean    = torch.zeros(64,           dtype=torch.float32, device='cuda')
    b_bn1_running_var     = torch.ones(64,            dtype=torch.float32, device='cuda')
    b_bn2_running_mean    = torch.zeros(64,           dtype=torch.float32, device='cuda')
    b_bn2_running_var     = torch.ones(64,            dtype=torch.float32, device='cuda')
    b_downsample_1_running_mean = torch.zeros(64,     dtype=torch.float32, device='cuda')
    b_downsample_1_running_var  = torch.ones(64,      dtype=torch.float32, device='cuda')
    x = torch.randn(10, 3, 224, 224,                  dtype=torch.float32, device='cuda')
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

        # Reference: PyTorch eager
        conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False).cuda()
        conv1.weight.data.copy_(p_conv1_weight)
        bn1 = nn.BatchNorm2d(64).cuda()
        bn1.weight.data.copy_(p_bn1_weight)
        bn1.bias.data.copy_(p_bn1_bias)
        bn1.running_mean.copy_(b_bn1_running_mean)
        bn1.running_var.copy_(b_bn1_running_var)

        conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False).cuda()
        conv2.weight.data.copy_(p_conv2_weight)
        bn2 = nn.BatchNorm2d(64).cuda()
        bn2.weight.data.copy_(p_bn2_weight)
        bn2.bias.data.copy_(p_bn2_bias)
        bn2.running_mean.copy_(b_bn2_running_mean)
        bn2.running_var.copy_(b_bn2_running_var)

        conv_ds = nn.Conv2d(3, 64, 1, stride=1, padding=0, bias=False).cuda()
        conv_ds.weight.data.copy_(p_downsample_0_weight)
        bn_ds = nn.BatchNorm2d(64).cuda()
        bn_ds.weight.data.copy_(p_downsample_1_weight)
        bn_ds.bias.data.copy_(p_downsample_1_bias)
        bn_ds.running_mean.copy_(b_downsample_1_running_mean)
        bn_ds.running_var.copy_(b_downsample_1_running_var)

        with torch.no_grad():
            identity = x.clone()
            out = conv1(x)
            out = bn1(out)
            out = torch.relu(out)
            out = conv2(out)
            out = bn2(out)
            identity = conv_ds(identity)
            identity = bn_ds(identity)
            out = out + identity
            ref = torch.relu(out)

        torch.cuda.synchronize()
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")