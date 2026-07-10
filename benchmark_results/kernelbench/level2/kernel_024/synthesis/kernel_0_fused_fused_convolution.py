import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['BATCH', 'OUT_C', 'OUT_D', 'OUT_H', 'OUT_W', 'IN_C'],
)
@triton.jit
def conv3d_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_D: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_D: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    stride_xn: tl.constexpr, stride_xc: tl.constexpr, stride_xd: tl.constexpr, stride_xh: tl.constexpr, stride_xw: tl.constexpr,
    stride_wco: tl.constexpr, stride_wci: tl.constexpr, stride_wd: tl.constexpr, stride_wh: tl.constexpr, stride_ww: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_D * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode m -> (batch, out_d, out_h, out_w)
    tmp = offs_m
    out_w_idx = tmp % OUT_W
    tmp = tmp // OUT_W
    out_h_idx = tmp % OUT_H
    tmp = tmp // OUT_H
    out_d_idx = tmp % OUT_D
    batch_idx = tmp // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KD * KH * KW
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_block in range(BLOCK_K_COUNT):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

        # Decode k -> (in_c, kd, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        tmp_k = tmp_k // KH
        kd_idx = tmp_k % KD
        ic_idx = tmp_k // KD

        # Input coordinates
        in_d_coord = out_d_idx[:, None] + kd_idx[None, :]
        in_h_coord = out_h_idx[:, None] + kh_idx[None, :]
        in_w_coord = out_w_idx[:, None] + kw_idx[None, :]

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_d = (in_d_coord >= 0) & (in_d_coord < IN_D)
        valid_h = (in_h_coord >= 0) & (in_h_coord < IN_H)
        valid_w = (in_w_coord >= 0) & (in_w_coord < IN_W)
        valid_ic = ic_idx[None, :] < IN_C

        x_mask = valid_m & valid_k & valid_d & valid_h & valid_w & valid_ic

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * stride_xn +
                  ic_idx[None, :] * stride_xc +

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Weight: [OUT_C, IN_C, KD, KH, KW]
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr +
                  offs_n[None, :] * stride_wco +
                  ic_idx[:, None] * stride_wci +
                  kd_idx[:, None] * stride_wd +
                  kh_idx[:, None] * stride_wh +
                  kw_idx[:, None] * stride_ww)

        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # Store output
    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)

    # Output shape: (BATCH, OUT_C, OUT_D, OUT_H, OUT_W)
    # stride: (OUT_C*OUT_D*OUT_H*OUT_W, OUT_D*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1)
    out_ptrs = (Out_ptr +
                batch_idx[:, None] * (OUT_C * OUT_D * OUT_H * OUT_W) +
                offs_n[None, :] * (OUT_D * OUT_H * OUT_W) +
                out_d_idx[:, None] * (OUT_H * OUT_W) +
                out_h_idx[:, None] * OUT_W +
                out_w_idx[:, None])

    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 2}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 4}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 8}, num_warps=4, num_stages=1),
    ],
    key=['xnumel', 'rnumel'],
)
@triton.jit
def min_dim2_kernel(
    Conv_ptr, Out_ptr,
    xnumel, rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr = 32,
):
    # Each program handles XBLOCK output elements
    # Output shape: (128, 24, 30, 30) = 2764800 elements
    # Reduction over dim=2 (OUT_D=22)
    # Input shape: (128, 24, 22, 30, 30)
    # Input stride: (475200, 19800, 900, 30, 1)
    # Output stride: (21600, 900, 30, 1)

    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel

    # xindex maps to (batch, out_c, out_h, out_w)
    x0 = xindex % 900          # out_h * 30 + out_w
    x4 = xindex // 900         # batch * 24 + out_c

    r_idx = tl.arange(0, RBLOCK)
    r_mask = r_idx < rnumel

    # Load all depth slices and find min
    # Input ptr offset: x0 + 900*r + 19800*x4
    in_ptrs = Conv_ptr + x0[:, None] + 900 * r_idx[None, :] + 19800 * x4[:, None]
    in_mask = xmask[:, None] & r_mask[None, :]
    vals = tl.load(in_ptrs, mask=in_mask, other=float('inf'))

    min_val = tl.min(vals, axis=1)

    tl.store(Out_ptr + xindex, min_val, mask=xmask)


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 2}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 4}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8}, num_warps=4, num_stages=1),
    ],
    key=['xnumel', 'rnumel'],
)
@triton.jit
def softmax_dim1_kernel(
    In_ptr, Out_ptr,
    xnumel, rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr = 32,
):
    # Softmax along dim=1 (channels=24)
    # Input shape: (128, 24, 30, 30)
    # xnumel = 128 * 30 * 30 = 115200 (batch * h * w)
    # rnumel = 24 (channels)

    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel

    # xindex -> (batch, h, w)
    x0 = xindex % 900   # h*30 + w
    x1 = xindex // 900  # batch

    r_idx = tl.arange(0, RBLOCK)
    r_mask = r_idx < rnumel

    # Load all channels: input[batch, c, h, w] = In_ptr[x0 + 900*c + 21600*x1]
    in_ptrs = In_ptr + x0[:, None] + 900 * r_idx[None, :] + 21600 * x1[:, None]
    in_mask = xmask[:, None] & r_mask[None, :]
    vals = tl.load(in_ptrs, mask=in_mask, other=0.0)

    # Numerically stable softmax
    vals_for_max = tl.where(r_mask[None, :], vals, float('-inf'))
    row_max = tl.max(vals_for_max, axis=1)

    shifted = vals - row_max[:, None]
    exp_vals = tl.exp(shifted)
    exp_vals_masked = tl.where(r_mask[None, :], exp_vals, 0.0)
    row_sum = tl.sum(exp_vals_masked, axis=1)

    softmax_vals = exp_vals / row_sum[:, None]

    out_ptrs = Out_ptr + x0[:, None] + 900 * r_idx[None, :] + 21600 * x1[:, None]
    tl.store(out_ptrs, softmax_vals, mask=in_mask)


def kernel_function(p_conv_weight, p_conv_bias, x):
    BATCH, IN_C, IN_D, IN_H, IN_W = x.shape
    OUT_C, _, KD, KH, KW = p_conv_weight.shape
    OUT_D = IN_D - KD + 1
    OUT_H = IN_H - KH + 1
    OUT_W = IN_W - KW + 1

    # Allocate conv output
    conv_out = torch.empty((BATCH, OUT_C, OUT_D, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_D * OUT_H * OUT_W

    BLOCK_M = 128
    BLOCK_N = 32

    grid_conv = (
        (M_TOTAL + BLOCK_M - 1) // BLOCK_M,
        (OUT_C + BLOCK_N - 1) // BLOCK_N,
    )

    conv3d_kernel[grid_conv](
        x, p_conv_weight, p_conv_bias, conv_out,
        BATCH, IN_C, IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W,
        KD, KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        p_conv_weight.stride(0), p_conv_weight.stride(1), p_conv_weight.stride(2),
        p_conv_weight.stride(3), p_conv_weight.stride(4),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=32,
    )

    # Min along dim=2 (depth)
    xnumel = BATCH * OUT_C * OUT_H * OUT_W
    rnumel = OUT_D
    min_out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    XBLOCK = 1
    grid_min = ((xnumel + XBLOCK - 1) // XBLOCK,)

    min_dim2_kernel[grid_min](
        conv_out, min_out,
        xnumel, rnumel,
        XBLOCK=XBLOCK, RBLOCK=32,
    )

    # Softmax along dim=1 (channels)
    xnumel_sm = BATCH * OUT_H * OUT_W
    rnumel_sm = OUT_C
    softmax_out = torch.empty_like(min_out)

    XBLOCK_SM = 1
    grid_sm = ((xnumel_sm + XBLOCK_SM - 1) // XBLOCK_SM,)

    softmax_dim1_kernel[grid_sm](
        min_out, softmax_out,
        xnumel_sm, rnumel_sm,
        XBLOCK=XBLOCK_SM, RBLOCK=32,
    )

    return softmax_out


def get_inputs():
    p_conv_weight = torch.randn(24, 3, 3, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias = torch.randn(24, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 3, 24, 32, 32, device='cuda', dtype=torch.float32)
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
        p_conv_weight, p_conv_bias, x = inputs
        import torch.nn as nn
        conv = nn.Conv3d(3, 24, 3, bias=True).cuda()
        with torch.no_grad():
            conv.weight.copy_(p_conv_weight)
            conv.bias.copy_(p_conv_bias)
        with torch.no_grad():
            ref_conv = conv(x)
            ref_min = torch.min(ref_conv, dim=2)[0]
            ref_softmax = torch.softmax(ref_min, dim=1)

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref_softmax.shape}")
        max_diff = (output - ref_softmax).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref_softmax, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")