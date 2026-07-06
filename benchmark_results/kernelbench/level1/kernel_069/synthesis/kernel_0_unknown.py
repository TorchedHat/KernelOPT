import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['M_TOTAL', 'CO', 'CI', 'KH', 'KW'],
)
@triton.jit
def conv_transpose2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, CI, H_IN, W_IN,
    CO, KH, KW,
    H_OUT, W_OUT,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # X strides (NCHW)
    s_xn, s_xc, s_xh, s_xw,
    # W strides (CI, CO, KH, KW)
    s_wci, s_wco, s_wkh, s_wkw,
    # Y strides (NCHW)
    s_yn, s_yco, s_yh, s_yw,
    M_TOTAL,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    
      Y[n, co, oh, ow] = sum_{ci, kh, kw} X[n, ci, ih, iw] * W[ci, co, kh, kw]
      where ih = (oh + PAD_H - kh) / STRIDE_H (only if divisible and in bounds)
            iw = (ow + PAD_W - kw) / STRIDE_W (only if divisible and in bounds)
    
    M = N * H_OUT * W_OUT (output spatial)
    N_dim = CO
    K = CI * KH * KW
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % W_OUT
    out_h = (offs_m // W_OUT) % H_OUT
    batch = offs_m // (W_OUT * H_OUT)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = CI * KH * KW

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K -> (ci, kh, kw)
        # K layout: ci * KH * KW + kh * KW + kw
        kw_idx = offs_k % KW
        kh_idx = (offs_k // KW) % KH
        ci_idx = offs_k // (KW * KH)

        # For ConvTranspose2d:
        # oh = ih * STRIDE_H - PAD_H + kh
        # => ih = (oh + PAD_H - kh) / STRIDE_H  (must be integer and in [0, H_IN))
        # Similarly for width

        oh_val = out_h  # shape [BLOCK_M]
        ow_val = out_w  # shape [BLOCK_M]

        # Compute numerator for ih and iw
        # ih_num[m, k] = oh[m] + PAD_H - kh[k]
        ih_num = oh_val[:, None] + PAD_H - kh_idx[None, :]  # [BLOCK_M, BLOCK_K]
        iw_num = ow_val[:, None] + PAD_W - kw_idx[None, :]  # [BLOCK_M, BLOCK_K]

        # Check divisibility by stride
        ih_div = ih_num // STRIDE_H
        iw_div = iw_num // STRIDE_W

        # Valid mask
        valid_k = offs_k < TOTAL_K
        valid_m = offs_m < M_TOTAL
        valid_batch = batch < N

        ih_valid = (ih_num % STRIDE_H == 0) & (ih_div >= 0) & (ih_div < H_IN)
        iw_valid = (iw_num % STRIDE_W == 0) & (iw_div >= 0) & (iw_div < W_IN)
        ci_valid = ci_idx < CI

        mask_x = (valid_m[:, None] & valid_batch[:, None] &

        # Load X tile [BLOCK_M, BLOCK_K]
        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Load W tile [BLOCK_K, BLOCK_N]
        # W shape: (CI, CO, KH, KW)
        mask_w = (valid_k[:, None] & ci_valid[:, None] &
                  (offs_n[None, :] < CO))
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Store output Y[n, co, oh, ow]
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < CO) & (batch[:, None] < N)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def kernel_function(weight, x):
    """
    """
    N, CI, H_IN, W_IN = x.shape
    # ConvTranspose2d weight shape is (in_channels, out_channels, kH, kW)
    CI_w, CO, KH, KW = weight.shape

    STRIDE_H = 1
    STRIDE_W = 1
    PAD_H = 0
    PAD_W = 0

    H_OUT = (H_IN - 1) * STRIDE_H - 2 * PAD_H + KH
    W_OUT = (W_IN - 1) * STRIDE_W - 2 * PAD_W + KW

    M_TOTAL = N * H_OUT * W_OUT

    y = torch.zeros(N, CO, H_OUT, W_OUT, device=x.device, dtype=x.dtype)

    # Make contiguous
    x_c = x.contiguous()
    w_c = weight.contiguous()

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(CO, meta['BLOCK_N']),
    )

    conv_transpose2d_kernel[grid](
        x_c, w_c, y,
        N, CI, H_IN, W_IN,
        CO, KH, KW,
        H_OUT, W_OUT,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        # X strides
        x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
        # W strides
        w_c.stride(0), w_c.stride(1), w_c.stride(2), w_c.stride(3),
        # Y strides
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        M_TOTAL,
    )

    return y


def get_inputs():
    weight = torch.randn(64, 128, 3, 5, dtype=torch.float32, device='cuda')
    x = torch.randn(64, 64, 128, 256, dtype=torch.float32, device='cuda')
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify against PyTorch reference
        weight_ref = inputs[0]
        x_ref = inputs[1]
        import torch.nn.functional as F
        ref = F.conv_transpose2d(x_ref, weight_ref, bias=None, stride=1, padding=0)
        print(f"Output shape: {output.shape}")
        print(f"Reference shape: {ref.shape}")
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference: {max_diff}")
        if max_diff < 1e-2:
            print("PASSED!")
        else:
            print("FAILED!")