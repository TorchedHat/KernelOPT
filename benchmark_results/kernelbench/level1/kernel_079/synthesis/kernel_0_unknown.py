import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'IN_L', 'OUT_L', 'KERNEL_SIZE'],
)
@triton.jit
def conv_transpose1d_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH, IN_C, OUT_C, IN_L, OUT_L,
    KERNEL_SIZE: tl.constexpr,
    STRIDE, PADDING, DILATION,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles a tile of (batch*out_pos, out_channel)
    # M = BATCH * OUT_L, N = OUT_C
    # For each output (b, oc, o_pos):
    #   Y[b, oc, o_pos] = sum over ic, k: X[b, ic, i_pos] * W[ic, oc, k]
    #   where i_pos = (o_pos + padding - k*dilation) / stride (if divisible and valid)

    pid_m = tl.program_id(0)  # tile over batch * out_pos
    pid_n = tl.program_id(1)  # tile over out_channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode offs_m -> (batch, out_pos)
    batch_idx = offs_m // OUT_L
    out_pos = offs_m % OUT_L

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    M_TOTAL = BATCH * OUT_L

    # Loop over in_channels in blocks of BLOCK_K
    for k_start in range(0, IN_C, BLOCK_K):
        ic_offs = k_start + tl.arange(0, BLOCK_K)
        ic_valid = ic_offs < IN_C

        # For each kernel position k (0..KERNEL_SIZE-1), accumulate
        for kk in range(KERNEL_SIZE):
            # For output position out_pos, kernel position kk:
            # i_pos = (out_pos + padding - kk * dilation) / stride
            # valid if (out_pos + padding - kk*dilation) % stride == 0 and 0 <= i_pos < IN_L
            shifted = out_pos + PADDING - kk * DILATION
            # Check divisibility and bounds
            rem = shifted % STRIDE
            i_pos = shifted // STRIDE

            valid_m = (rem == 0) & (i_pos >= 0) & (i_pos < IN_L) & (batch_idx < BATCH)

            # Load X tile: shape [BLOCK_M, BLOCK_K]
            # X[batch_idx, ic, i_pos] with stride: batch*IN_C*IN_L, ic*IN_L, i_pos
            x_ptrs = (X_ptr
            x_mask = valid_m[:, None] & ic_valid[None, :]
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # Load W tile: shape [BLOCK_K, BLOCK_N]
            # W[ic, oc, kk] with stride: ic*OUT_C*KERNEL_SIZE, oc*KERNEL_SIZE, kk
            w_ptrs = (W_ptr
            w_mask = ic_valid[:, None] & (offs_n[None, :] < OUT_C)
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

            acc = tl.dot(x_tile, w_tile, acc)

    # Store output
    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def kernel_function(weight, x):
    # weight: (in_channels, out_channels, kernel_size) = (32, 64, 3)
    # x: (batch, in_channels, in_length) = (16, 32, 131072)
    # ConvTranspose1d params: stride=2, padding=1, dilation=2
    STRIDE = 2
    PADDING = 1
    DILATION = 2

    batch, in_c, in_l = x.shape
    in_c2, out_c, kernel_size = weight.shape
    assert in_c == in_c2

    out_l = (in_l - 1) * STRIDE - 2 * PADDING + DILATION * (kernel_size - 1) + 1

    y = torch.zeros(batch, out_c, out_l, device=x.device, dtype=x.dtype)

    M_TOTAL = batch * out_l
    N_TOTAL = out_c

    x_contig = x.contiguous()
    w_contig = weight.contiguous()

    def grid(meta):
        return (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(N_TOTAL, meta['BLOCK_N']),
        )

    conv_transpose1d_kernel[grid](
        x_contig, w_contig, y,
        batch, in_c, out_c, in_l, out_l,
        kernel_size,
        STRIDE, PADDING, DILATION,
    )

    return y


def get_inputs():
    weight = torch.randn(32, 64, 3, dtype=torch.float32, device='cuda')
    x = torch.randn(16, 32, 131072, dtype=torch.float32, device='cuda')
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    weight, x = inputs

    # Verify correctness
    if not args.profile_only:
        import torch.nn as nn
        conv = nn.ConvTranspose1d(32, 64, 3, stride=2, padding=1, dilation=2, bias=False)
        conv.weight.data = weight.cpu()
        conv = conv.cuda()
        with torch.no_grad():
            ref = conv(x)

        out = kernel_function(weight, x)
        torch.cuda.synchronize()

        max_diff = (out - ref).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Correctness check passed!")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")