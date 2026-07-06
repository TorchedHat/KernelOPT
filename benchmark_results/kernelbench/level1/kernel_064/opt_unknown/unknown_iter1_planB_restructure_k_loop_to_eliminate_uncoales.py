import torch
import triton
import triton.language as tl

# ConvTranspose1d: weight shape [in_channels, out_channels, kernel_size]
# Input: [batch, in_channels, L_in]
# Output: [batch, out_channels, L_out] where L_out = L_in + kernel_size - 1 (stride=1, padding=0)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose1d_kernel(
    X_ptr, W_ptr, Y_ptr,
    batch, in_channels, out_channels, L_in, L_out, kernel_size,
    stride_xb, stride_xc, stride_xl,
    stride_wb, stride_wc, stride_wk,
    stride_yb, stride_yc, stride_yl,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (batch, cdiv(L_out, BLOCK_M), cdiv(out_channels, BLOCK_N))
    # M = L_out, N = out_channels, K = in_channels * kernel_size
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output positions
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channels

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ConvTranspose1d: Y[b, oc, l_out] = sum_{ic, kk} X[b, ic, l_out - kk] * W[ic, oc, kk]
    # Restructured to nested loops: outer over kk (scalar), inner over ic (contiguous block)
    # This eliminates in-loop // and % operations and makes memory accesses coalesced.

    valid_m = offs_m < M
    valid_n = offs_n < N

    for kk_idx in range(kernel_size):  # outer: scalar kernel position
        l_in = offs_m - kk_idx          # [BLOCK_M]
        valid_l_in = (l_in >= 0) & (l_in < L_in)  # [BLOCK_M]
        for ic_start in range(0, in_channels, BLOCK_K):  # inner: block over channels
            offs_ic = ic_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            valid_ic = offs_ic < in_channels
            # X: [BLOCK_M, BLOCK_K] - ic contiguous along cols, l_in broadcast along rows
            mask_x = (valid_m & valid_l_in)[:, None] & valid_ic[None, :]
            x_ptrs = (X_ptr + pid_b * stride_xb
                      + offs_ic[None, :] * stride_xc
                      + l_in[:, None] * stride_xl)
            x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)  # [BLOCK_M, BLOCK_K]
            # W: [BLOCK_K, BLOCK_N] - ic contiguous along rows, oc contiguous along cols
            mask_w = valid_ic[:, None] & valid_n[None, :]
            w_ptrs = (W_ptr
                      + offs_ic[:, None] * stride_wb
                      + offs_n[None, :] * stride_wc
                      + kk_idx * stride_wk)  # kk scalar offset
            w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)  # [BLOCK_K, BLOCK_N]
            acc = tl.dot(x_tile, w_tile, acc=acc)

    # Store output Y[b, oc, l_out]
    mask_y = valid_m[:, None] & valid_n[None, :]
    y_ptrs = Y_ptr + pid_b * stride_yb + offs_n[None, :] * stride_yc + offs_m[:, None] * stride_yl
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_y)

def kernel_function(weight, x):
    # weight: [in_channels, out_channels, kernel_size]
    # x: [batch, in_channels, L_in]
    batch, in_channels, L_in = x.shape
    in_ch_w, out_channels, kernel_size = weight.shape

    stride = 1
    padding = 0
    L_out = (L_in - 1) * stride - 2 * padding + kernel_size  # = L_in + kernel_size - 1

    y = torch.zeros(batch, out_channels, L_out, device=x.device, dtype=x.dtype)

    M = L_out
    N = out_channels
    K = in_channels * kernel_size

    stride_xb, stride_xc, stride_xl = x.stride()
    stride_wb, stride_wc, stride_wk = weight.stride()
    stride_yb, stride_yc, stride_yl = y.stride()

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (batch, cdiv(M, meta['BLOCK_M']), cdiv(N, meta['BLOCK_N']))

    conv_transpose1d_kernel[grid](
        x, weight, y,
        batch, in_channels, out_channels, L_in, L_out, kernel_size,
        stride_xb, stride_xc, stride_xl,
        stride_wb, stride_wc, stride_wk,
        stride_yb, stride_yc, stride_yl,
        M, N, K,
    )

    return y


def get_inputs():
    batch_size = 64
    in_channels = 128
    out_channels = 128
    kernel_size = 3
    length = 65536

    weight = torch.randn(in_channels, out_channels, kernel_size, device='cuda', dtype=torch.float32)
    x = torch.rand(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
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
        conv = nn.ConvTranspose1d(128, 128, 3, bias=False).cuda()
        conv.weight.data.copy_(weight)
        ref = conv(x)
        out = kernel_function(weight, x)
        print(f"Max diff: {(out - ref).abs().max().item()}")
        print(f"Output shape: {out.shape}, ref shape: {ref.shape}")
        assert torch.allclose(out, ref, atol=1e-3), "Mismatch!"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")