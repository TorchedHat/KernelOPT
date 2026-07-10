import torch
import triton
import triton.language as tl

# ConvTranspose3d kernel: implicit GEMM approach
# For transposed conv: output[n, co, od, oh, ow] = sum over ci,kd,kh,kw of
#   input[n, ci, id, ih, iw] * weight[ci, co, kd, kh, kw]
# where id = (od + padding - kd) / stride (only if divisible)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, CI, CO,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    M, N_out, K,
    # X strides: (N, CI, ID, IH, IW)
    sx_n, sx_ci, sx_d, sx_h, sx_w,
    # W strides: (CI, CO, KD, KH, KW)
    sw_ci, sw_co, sw_kd, sw_kh, sw_kw,
    # Y strides: (N, CO, OD, OH, OW)
    sy_n, sy_co, sy_d, sy_h, sy_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = N * OD * OH * OW (output spatial positions)
    # N_out = CO
    # K = CI * KD * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, od, oh, ow)
    OHW = OH * OW
    ODH = OD * OH * OW
    batch = offs_m // ODH
    rem = offs_m % ODH
    od = rem // OHW
    rem2 = rem % OHW
    oh = rem2 // OW
    ow = rem2 % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (ci, kd, kh, kw)
        KHKW = KH * KW
        KDKHKW = KD * KH * KW
        ci = offs_k // KDKHKW
        rem_k = offs_k % KDKHKW
        kd = rem_k // KHKW
        rem_k2 = rem_k % KHKW
        kh = rem_k2 // KW
        kw = rem_k2 % KW

        # For transposed conv: input position id = (od + PAD_D - kd) / STRIDE_D
        # Only valid if (od + PAD_D - kd) is divisible by STRIDE_D and in range
        od_ext = od[:, None] + PAD_D - kd[None, :]  # (BLOCK_M, BLOCK_K)
        oh_ext = oh[:, None] + PAD_H - kh[None, :]
        ow_ext = ow[:, None] + PAD_W - kw[None, :]

        id_val = od_ext // STRIDE_D
        ih_val = oh_ext // STRIDE_H
        iw_val = ow_ext // STRIDE_W

        valid_d = (od_ext % STRIDE_D == 0) & (id_val >= 0) & (id_val < ID)
        valid_h = (oh_ext % STRIDE_H == 0) & (ih_val >= 0) & (ih_val < IH)
        valid_w = (ow_ext % STRIDE_W == 0) & (iw_val >= 0) & (iw_val < IW)
        valid_ci = ci[None, :] < CI
        valid_k = offs_k[None, :] < K
        valid_batch = batch[:, None] < N

        valid_x = valid_d & valid_h & valid_w & valid_ci & valid_k & valid_batch

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: W[ci, co, kd, kh, kw]
        valid_w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < CO)
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=valid_w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < CO, other=0.0)
    acc = acc + bias[None, :]

    # Store output
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < CO)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def maxpool_maxpool_sum_kernel(
    X_ptr, Y_ptr,
    N, C,
    # Input dims after conv_transpose: (OD, OH, OW)
    ID, IH, IW,
    # After pool1 (kernel=2, stride=2): P1D, P1H, P1W
    P1D, P1H, P1W,
    # After pool2 (kernel=3, stride=3): P2D, P2H, P2W
    P2D, P2H, P2W,
    total_elements,
    # X strides (N, C, ID, IH, IW)
    sx_n, sx_c, sx_d, sx_h, sx_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Output: (N, 1, P2D, P2H, P2W)
    # Each output element = sum over C of max_pool2(max_pool1(X))[n, c, d2, h2, w2]
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    # Decode output index: (n, d2, h2, w2) - summed over C
    P2HW = P2H * P2W
    P2DHW = P2D * P2H * P2W
    n = offs // P2DHW
    rem = offs % P2DHW
    d2 = rem // P2HW
    rem2 = rem % P2HW
    h2 = rem2 // P2W
    w2 = rem2 % P2W

    # Pool2 window in pool1 space: d2*3 to d2*3+3, etc.
    # Pool1 window in input space: p1_d*2 to p1_d*2+2, etc.
    # Combined: input range for pool2[d2] = [d2*3*2, d2*3*2+6) = [d2*6, d2*6+6)

    result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for c in range(C):
        # Max over pool2 window (3x3x3 in pool1 space, each pool1 is 2x2x2 in input)
        # Total: 6x6x6 window in input space
        chan_max = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

        for kd2 in range(3):
            for kh2 in range(3):
                for kw2 in range(3):
                    # Pool1 position
                    p1d = d2 * 3 + kd2
                    p1h = h2 * 3 + kh2
                    p1w = w2 * 3 + kw2

                    # Max over pool1 window (2x2x2 in input)
                    pool1_max = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
                    for kd1 in range(2):
                        for kh1 in range(2):
                            for kw1 in range(2):
                                id_val = p1d * 2 + kd1
                                ih_val = p1h * 2 + kh1
                                iw_val = p1w * 2 + kw1

                                valid = mask & (n < N) & (id_val < ID) & (ih_val < IH) & (iw_val < IW)
                                x_ptrs = (X_ptr
                                val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
                                pool1_max = tl.maximum(pool1_max, val)

                    chan_max = tl.maximum(chan_max, pool1_max)

        result = result + chan_max

    # Store result
    out_ptrs = Y_ptr + offs
    tl.store(out_ptrs, result, mask=mask)


def kernel_function(p_conv_transpose_weight, p_conv_transpose_bias, x):
    # x: (N, CI, ID, IH, IW) = (16, 32, 32, 32, 32)
    # weight: (CI, CO, KD, KH, KW) = (32, 64, 5, 5, 5)
    # bias: (CO,) = (64,)

    N, CI, ID, IH, IW = x.shape
    CI_w, CO, KD, KH, KW = p_conv_transpose_weight.shape
    STRIDE_D = STRIDE_H = STRIDE_W = 2
    PAD_D = PAD_H = PAD_W = 2

    # Output size for transposed conv:
    # OD = (ID - 1) * STRIDE_D - 2 * PAD_D + KD = (32-1)*2 - 4 + 5 = 62 - 4 + 5 = 63
    OD = (ID - 1) * STRIDE_D - 2 * PAD_D + KD
    OH = (IH - 1) * STRIDE_H - 2 * PAD_H + KH
    OW = (IW - 1) * STRIDE_W - 2 * PAD_W + KW

    # Allocate conv output
    conv_out = torch.empty((N, CO, OD, OH, OW), device=x.device, dtype=x.dtype)

    M = N * OD * OH * OW
    N_out = CO
    K = CI * KD * KH * KW

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_out, meta['BLOCK_N']))

    conv_transpose3d_kernel[grid](
        x, p_conv_transpose_weight, p_conv_transpose_bias, conv_out,
        N, CI, CO,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        STRIDE_D, STRIDE_H, STRIDE_W,
        PAD_D, PAD_H, PAD_W,
        M, N_out, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        p_conv_transpose_weight.stride(0), p_conv_transpose_weight.stride(1),
        p_conv_transpose_weight.stride(2), p_conv_transpose_weight.stride(3),
        p_conv_transpose_weight.stride(4),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        conv_out.stride(3), conv_out.stride(4),
    )

    # MaxPool3d(2) -> (N, CO, 31, 31, 31)
    P1D = OD // 2  # 31
    P1H = OH // 2  # 31
    P1W = OW // 2  # 31

    # MaxPool3d(3) -> (N, CO, 10, 10, 10)
    P2D = P1D // 3  # 10
    P2H = P1H // 3  # 10
    P2W = P1W // 3  # 10

    # Output: (N, 1, P2D, P2H, P2W)
    out = torch.empty((N, 1, P2D, P2H, P2W), device=x.device, dtype=x.dtype)
    total_elements = N * P2D * P2H * P2W

    grid2 = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)

    maxpool_maxpool_sum_kernel[grid2](
        conv_out, out,
        N, CO,
        OD, OH, OW,
        P1D, P1H, P1W,
        P2D, P2H, P2W,
        total_elements,
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        conv_out.stride(3), conv_out.stride(4),
    )

    return out


def get_inputs():
    p_conv_transpose_weight = torch.randn(32, 64, 5, 5, 5)
    p_conv_transpose_bias = torch.randn(64)
    x = torch.rand(16, 32, 32, 32, 32)
    return [p_conv_transpose_weight, p_conv_transpose_bias, x]


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
        w, b, x = inputs
        w_cuda = w.cuda()
        b_cuda = b.cuda()
        x_cuda = x.cuda()

        # Reference computation
        conv_t = nn.ConvTranspose3d(32, 64, 5, stride=2, padding=2).cuda()
        conv_t.weight.data = w_cuda
        conv_t.bias.data = b_cuda

        with torch.no_grad():
            ref = conv_t(x_cuda)
            ref = nn.MaxPool3d(kernel_size=2)(ref)
            ref = nn.MaxPool3d(kernel_size=3)(ref)
            ref = torch.sum(ref, dim=1, keepdim=True)

        print(f"Output shape: {output.shape}, ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={(output - ref).abs().max().item():.4e}"
        print("Kernel executed successfully.")