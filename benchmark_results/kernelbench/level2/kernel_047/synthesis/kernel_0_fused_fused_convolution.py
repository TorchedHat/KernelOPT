import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ── Conv3d implicit-GEMM + bias + mish + tanh ──────────────────────────────
# M = Batch * OUT_D * OUT_H * OUT_W
# N = OUT_C
# K = IN_C * KD * KH * KW

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128,'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128,'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_D', 'OUT_H', 'OUT_W',
         'KD', 'KH', 'KW'],
)
@triton.jit
def conv3d_mish_tanh_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr,
    IN_D: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_D: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_D: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # strides for X: (BATCH, IN_C, IN_D, IN_H, IN_W)
    sXn: tl.constexpr, sXc: tl.constexpr, sXd: tl.constexpr, sXh: tl.constexpr, sXw: tl.constexpr,
    # strides for W: (OUT_C, IN_C, KD, KH, KW)
    sWo: tl.constexpr, sWi: tl.constexpr, sWd: tl.constexpr, sWh: tl.constexpr, sWw: tl.constexpr,
    # strides for Y: (BATCH, OUT_C, OUT_D, OUT_H, OUT_W)
    sYn: tl.constexpr, sYc: tl.constexpr, sYd: tl.constexpr, sYh: tl.constexpr, sYw: tl.constexpr,
    M_TOTAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_d, out_h, out_w)
    tmp = offs_m
    out_w = tmp % OUT_W
    tmp = tmp // OUT_W
    out_h = tmp % OUT_H
    tmp = tmp // OUT_H
    out_d = tmp % OUT_D
    batch  = tmp // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KD * KH * KW

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kd, kh, kw)
        tmp_k = offs_k
        in_c = tmp_k % IN_C
        tmp_k = tmp_k // IN_C
        kw = tmp_k % KW
        tmp_k = tmp_k // KW
        kh = tmp_k % KH
        kd = tmp_k // KH

        # Input coordinates
        id_ = out_d[:, None] * STRIDE_D + kd[None, :] - PAD_D
        ih  = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw  = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = (offs_m[:, None] < M_TOTAL) & (batch[:, None] < BATCH)
        valid_k = offs_k[None, :] < TOTAL_K
        valid_spatial = (
            (id_ >= 0) & (id_ < IN_D) &
            (ih  >= 0) & (ih  < IN_H) &
            (iw  >= 0) & (iw  < IN_W)
        )
        mask_x = valid_m & valid_k & valid_spatial

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight tile [BLOCK_K, BLOCK_N]
        mask_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # Mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x)), numerically: if x > 20, softplus ≈ x
    threshold = tl.full((1,), 20.0, dtype=tl.float32)
    gt20 = acc > threshold
    exp_acc = libdevice.exp(acc)
    sp = tl.where(gt20, acc, libdevice.log1p(exp_acc))
    mish_out = acc * libdevice.tanh(sp)

    # Tanh
    out = libdevice.tanh(mish_out)

    # Store
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, out.to(tl.float32), mask=mask_out)


def kernel_function(p_conv_weight, p_conv_bias, x):
    # x: (N, Ci, D, H, W)
    # w: (Co, Ci, KD, KH, KW)
    N, Ci, D, H, W = x.shape
    Co, _, KD, KH, KW = p_conv_weight.shape

    STRIDE_D, STRIDE_H, STRIDE_W = 1, 1, 1
    PAD_D, PAD_H, PAD_W = 0, 0, 0

    OUT_D = (D + 2 * PAD_D - KD) // STRIDE_D + 1
    OUT_H = (H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (W + 2 * PAD_W - KW) // STRIDE_W + 1

    M_TOTAL = N * OUT_D * OUT_H * OUT_W

    y = torch.empty((N, Co, OUT_D, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    # Strides
    sXn, sXc, sXd, sXh, sXw = x.stride()
    sWo, sWi, sWd, sWh, sWw = p_conv_weight.stride()
    sYn, sYc, sYd, sYh, sYw = y.stride()

    def cdiv(a, b): return (a + b - 1) // b

    grid = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(Co, meta['BLOCK_N']))

    conv3d_mish_tanh_kernel[grid](
        x, p_conv_weight, p_conv_bias, y,
        N, Ci,
        D, H, W,
        Co, OUT_D, OUT_H, OUT_W,
        KD, KH, KW,
        STRIDE_D, STRIDE_H, STRIDE_W,
        PAD_D, PAD_H, PAD_W,
        sXn, sXc, sXd, sXh, sXw,
        sWo, sWi, sWd, sWh, sWw,
        sYn, sYc, sYd, sYh, sYw,
        M_TOTAL,
    )
    return y


def get_inputs():
    p_conv_weight = torch.randn(64, 32, 3, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias   = torch.randn(64, device='cuda', dtype=torch.float32)
    x             = torch.rand(16, 32, 32, 64, 64, device='cuda', dtype=torch.float32)
    return [p_conv_weight, p_conv_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn.functional as F
        p_conv_weight, p_conv_bias, x = inputs
        # Reference
        conv_out = F.conv3d(x, p_conv_weight, p_conv_bias, stride=1, padding=0)
        # Mish: x * tanh(softplus(x))
        sp = torch.where(conv_out > 20, conv_out, torch.log1p(torch.exp(conv_out)))
        mish_out = conv_out * torch.tanh(sp)
        ref = torch.tanh(mish_out)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")