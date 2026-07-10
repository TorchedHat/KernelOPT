import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'IN_C'],
)
@triton.jit
def conv_transpose3d_clamp_div_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_D, IN_H, IN_W,
    OUT_C, OUT_D, OUT_H, OUT_W,
    KD, KH, KW,
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    M_TOTAL,
    min_val, divisor,
    # Input strides (NCDHW)
    s_xn, s_xc, s_xd, s_xh, s_xw,
    # Weight strides (in_c, out_c, kd, kh, kw)
    s_wi, s_wo, s_wkd, s_wkh, s_wkw,
    # Output strides (NCDHW)
    s_yn, s_yc, s_yd, s_yh, s_yw,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index -> (batch, out_d, out_h, out_w)
    tmp = offs_m
    out_w = tmp % OUT_W
    tmp = tmp // OUT_W
    out_h = tmp % OUT_H
    tmp = tmp // OUT_H
    out_d = tmp % OUT_D
    batch = tmp // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K dimension: IN_C * KD * KH * KW
    TOTAL_K = IN_C * KD * KH * KW

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K index -> (in_c, kd, kh, kw)
        tmp_k = offs_k
        kw_idx = tmp_k % KW
        tmp_k = tmp_k // KW
        kh_idx = tmp_k % KH
        tmp_k = tmp_k // KH
        kd_idx = tmp_k % KD
        in_c_idx = tmp_k // KD

        # For transposed conv: out = in * stride + k - pad
        # So: in = (out - k + pad) / stride  (must be integer and in bounds)
        # out_d[m] corresponds to in_d = (out_d - kd + pad) / stride
        od_minus_kd = out_d[:, None] - kd_idx[None, :] + PAD_D
        oh_minus_kh = out_h[:, None] - kh_idx[None, :] + PAD_H
        ow_minus_kw = out_w[:, None] - kw_idx[None, :] + PAD_W

        # Check divisibility by stride
        valid_d = (od_minus_kd % STRIDE_D) == 0
        valid_h = (oh_minus_kh % STRIDE_H) == 0
        valid_w = (ow_minus_kw % STRIDE_W) == 0

        in_d_idx = od_minus_kd // STRIDE_D
        in_h_idx = oh_minus_kh // STRIDE_H
        in_w_idx = ow_minus_kw // STRIDE_W

        # Bounds check
        in_d_valid = (in_d_idx >= 0) & (in_d_idx < IN_D)
        in_h_valid = (in_h_idx >= 0) & (in_h_idx < IN_H)
        in_w_valid = (in_w_idx >= 0) & (in_w_idx < IN_W)
        k_valid = offs_k[None, :] < TOTAL_K
        in_c_valid = in_c_idx[None, :] < IN_C

        valid_mask = valid_d & valid_h & valid_w & in_d_valid & in_h_valid & in_w_valid & k_valid & in_c_valid & (offs_m[:, None] < M_TOTAL) & (batch[:, None] < N)

        # Load input [BLOCK_M, BLOCK_K]
        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=valid_mask, other=0.0)

        # Load weight [BLOCK_K, BLOCK_N]
        # Weight shape: (IN_C, OUT_C, KD, KH, KW)
        n_valid = offs_n[None, :] < OUT_C
        w_ptrs = (W_ptr
        w_mask = (offs_k[:, None] < TOTAL_K) & n_valid & (in_c_idx[:, None] < IN_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    b_offs = offs_n
    b_mask = b_offs < OUT_C
    bias = tl.load(B_ptr + b_offs, mask=b_mask, other=0.0)
    acc = acc + bias[None, :]

    # Clamp and divide
    acc = tl.maximum(acc, min_val)
    acc = acc / divisor

    # Store output
    # Output layout: (N, OUT_C, OUT_D, OUT_H, OUT_W)
    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C) & (batch[:, None] < N)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def kernel_function(weight, bias, x):
    N, IN_C, IN_D, IN_H, IN_W = x.shape
    OUT_C, _, KD, KH, KW = weight.shape  # weight is (IN_C, OUT_C, KD, KH, KW) for ConvTranspose3d

    STRIDE_D, STRIDE_H, STRIDE_W = 2, 2, 2
    PAD_D, PAD_H, PAD_W = 1, 1, 1

    OUT_D = (IN_D - 1) * STRIDE_D - 2 * PAD_D + KD
    OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + KH
    OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + KW

    min_val = -1.0
    divisor = 2.0

    M_TOTAL = N * OUT_D * OUT_H * OUT_W

    y = torch.empty((N, OUT_C, OUT_D, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    # Weight for ConvTranspose3d in PyTorch: (in_channels, out_channels, kD, kH, kW)
    s_wi, s_wo, s_wkd, s_wkh, s_wkw = weight.stride()

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv_transpose3d_clamp_div_kernel[grid](
        x, weight, bias, y,
        N, IN_C, IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W,
        KD, KH, KW,
        STRIDE_D, STRIDE_H, STRIDE_W,
        PAD_D, PAD_H, PAD_W,
        M_TOTAL,
        min_val, divisor,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
    )

    return y


def get_inputs():
    weight = torch.randn(64, 128, 3, 3, 3)
    bias = torch.randn(128)
    x = torch.rand(16, 64, 24, 48, 48)
    return [weight, bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    # Move to GPU
    inputs = [t.cuda() for t in get_inputs()]
    weight, bias, x = inputs

    output = kernel_function(weight, bias, x)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn
        # Reference: ConvTranspose3d with same parameters
        conv = nn.ConvTranspose3d(64, 128, 3, stride=2, padding=1).cuda()
        with torch.no_grad():
            conv.weight.copy_(weight)
            conv.bias.copy_(bias)
            ref = conv(x)
            ref = torch.clamp(ref, min=-1.0)
            ref = ref / 2.0

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")