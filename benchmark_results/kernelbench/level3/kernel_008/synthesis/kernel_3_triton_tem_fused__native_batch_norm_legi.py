import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Implicit-GEMM convolution (generic)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
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
    # strides for W (OUT_C, IN_C, KH, KW) -> stored as (OUT_C, KH, KW, IN_C) for NHWC
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

        # Decode K -> (kh, kw, in_c)  [NHWC weight layout: OUT_C, KH, KW, IN_C]
        in_c = offs_k % IN_C
        kw   = (offs_k // IN_C) % KERNEL_W
        kh   = offs_k // (IN_C * KERNEL_W)

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m < M_TOTAL
        valid_k = offs_k < TOTAL_K
        valid = (
            valid_m[:, None]
            & (ih >= 0) & (ih < IN_H)
            & (iw >= 0) & (iw < IN_W)
        )

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # Weight: (OUT_C, KH, KW, IN_C) layout
        w_ptrs = (W_ptr
        valid_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Store output in NHWC layout
    valid_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Fused BN + ReLU (pointwise, NHWC input)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N_TOTAL', 'C'],
)
@triton.jit
def bn_relu_kernel(
    X_ptr, Y_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    N_TOTAL,  # total number of elements = B*H*W*C
    C: tl.constexpr,
    eps: tl.constexpr,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Channel index (NHWC: last dim is C)
    c_idx = offs % C

    mean   = tl.load(mean_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)
    var    = tl.load(var_ptr    + c_idx, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    bias   = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * weight + bias

    if do_relu:
        out = tl.maximum(out, 0.0)

    tl.store(Y_ptr + offs, out.to(tl.float32), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Fused BN + add + ReLU (pointwise, NHWC)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N_TOTAL', 'C'],
)
@triton.jit
def bn_add_relu_kernel(
    X_ptr, residual_ptr, Y_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    N_TOTAL,
    C: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    x   = tl.load(X_ptr        + offs, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr  + offs, mask=mask, other=0.0).to(tl.float32)

    c_idx = offs % C

    mean   = tl.load(mean_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)
    var    = tl.load(var_ptr    + c_idx, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    bias   = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * weight + bias + res

    out = tl.maximum(out, 0.0)

    tl.store(Y_ptr + offs, out.to(tl.float32), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one conv2d via implicit GEMM
# ─────────────────────────────────────────────────────────────────────────────
def run_conv2d(x_nhwc, w_nchw, out_h, out_w, stride_h, stride_w, pad_h, pad_w):
    """
    returns y_nhwc: (N, out_h, out_w, C_out)
    """
    N, H, W, C_in = x_nhwc.shape
    C_out, _, KH, KW = w_nchw.shape

    # Convert weight to NHWC-friendly layout: (C_out, KH, KW, C_in)
    w_ohwi = w_nchw.permute(0, 2, 3, 1).contiguous()

    M_TOTAL = N * out_h * out_w
    y_nhwc = torch.empty((N, out_h, out_w, C_out), device=x_nhwc.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(C_out,   meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_ohwi, y_nhwc,
        N, C_in, H, W,
        C_out, out_h, out_w,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        # X strides (NHWC)
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        # W strides (C_out, KH, KW, C_in)
        w_ohwi.stride(0), w_ohwi.stride(1), w_ohwi.stride(2), w_ohwi.stride(3),
        # Y strides (NHWC)
        y_nhwc.stride(0), y_nhwc.stride(1), y_nhwc.stride(2), y_nhwc.stride(3),
        M_TOTAL,
    )
    return y_nhwc


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel function
# ─────────────────────────────────────────────────────────────────────────────
def kernel_function(
):
    N, C_in, H, W = x.shape
    eps = 1e-5

    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()

    # ── conv1: 3x3, stride=1, pad=1, in_c=3, out_c=64 ──
    out1_nhwc = run_conv2d(x_nhwc, p_conv1_weight, H, W, 1, 1, 1, 1)

    # ── BN1 + ReLU (fused) ──
    N_TOTAL1 = N * H * W * 64
    bn1_out = torch.empty_like(out1_nhwc)
    grid1 = lambda meta: (triton.cdiv(N_TOTAL1, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid1](
        out1_nhwc, bn1_out,
        b_bn1_running_mean, b_bn1_running_var,
        p_bn1_weight, p_bn1_bias,
        N_TOTAL1, 64, eps, True,
    )

    # ── conv2: 3x3, stride=1, pad=1, in_c=64, out_c=64 ──
    out2_nhwc = run_conv2d(bn1_out, p_conv2_weight, H, W, 1, 1, 1, 1)

    # ── Downsample conv: 1x1, stride=1, pad=0, in_c=3, out_c=64 ──
    ds_nhwc = run_conv2d(x_nhwc, p_downsample_0_weight, H, W, 1, 1, 0, 0)

    # ── Downsample BN (no relu) ──
    N_TOTAL_ds = N * H * W * 64
    ds_bn_out = torch.empty_like(ds_nhwc)
    grid_ds = lambda meta: (triton.cdiv(N_TOTAL_ds, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid_ds](
        ds_nhwc, ds_bn_out,
        b_downsample_1_running_mean, b_downsample_1_running_var,
        p_downsample_1_weight, p_downsample_1_bias,
        N_TOTAL_ds, 64, eps, False,
    )

    # ── BN2 + add(residual) + ReLU (fused) ──
    N_TOTAL2 = N * H * W * 64
    final_out = torch.empty_like(out2_nhwc)
    grid2 = lambda meta: (triton.cdiv(N_TOTAL2, meta['BLOCK_SIZE']),)
    bn_add_relu_kernel[grid2](
        out2_nhwc, ds_bn_out, final_out,
        b_bn2_running_mean, b_bn2_running_var,
        p_bn2_weight, p_bn2_bias,
        N_TOTAL2, 64, eps,
    )

    # Convert back to NCHW
    return final_out.permute(0, 3, 1, 2).contiguous()


def get_inputs():
    in_channels = 3
    out_channels = 64
    stride = 1
    batch_size = 10

    import torch.nn as nn

    model = _make_model(in_channels, out_channels, stride)
    model.eval()

    x = torch.rand(batch_size, in_channels, 224, 224, device='cuda')

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    return [
        params['conv1.weight'].cuda(),
        params['bn1.weight'].cuda(),
        params['bn1.bias'].cuda(),
        params['conv2.weight'].cuda(),
        params['bn2.weight'].cuda(),
        params['bn2.bias'].cuda(),
        params['downsample.0.weight'].cuda(),
        params['downsample.1.weight'].cuda(),
        params['downsample.1.bias'].cuda(),
        buffers['bn1.running_mean'].cuda(),
        buffers['bn1.running_var'].cuda(),
        buffers['bn2.running_mean'].cuda(),
        buffers['bn2.running_var'].cuda(),
        buffers['downsample.1.running_mean'].cuda(),
        buffers['downsample.1.running_var'].cuda(),
        x,
    ]


def _make_model(in_channels, out_channels, stride):
    import torch.nn as nn

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_channels)
            self.relu  = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
            self.stride = stride

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out

    return BasicBlock(in_channels, out_channels, stride)


if __name__ == "__main__":
    import argparse
    import torch.nn as nn

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Reference
        in_channels = 3
        out_channels = 64
        stride = 1
        batch_size = 10

        model = _make_model(in_channels, out_channels, stride)
        model.eval().cuda()

        x = inputs[-1]
        with torch.no_grad():
            ref = model(x)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")