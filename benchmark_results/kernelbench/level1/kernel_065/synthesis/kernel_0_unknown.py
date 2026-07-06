import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M_TOTAL', 'OUT_C', 'IN_C', 'KH', 'KW'],
)
@triton.jit
def conv_transpose2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wci, stride_wco, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    M_TOTAL,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = IN_C * KH * KW
    TOTAL_K = IN_C * KH * KW

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K index -> (in_c, kh, kw)
        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        in_c = offs_k // (KH * KW)

        # For ConvTranspose2d: output[n, co, oh, ow] += input[n, ci, ih, iw] * weight[ci, co, kh, kw]
        # where oh = ih + kh (stride=1, pad=0), so ih = oh - kh
        ih = out_h[:, None] - kh[None, :]
        iw = out_w[:, None] - kw[None, :]

        valid_k = offs_k < TOTAL_K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        valid_m = offs_m < M_TOTAL
        valid_batch = batch < N

        mask_x = valid_m[:, None] & valid_batch[:, None] & valid_spatial & valid_k[None, :]

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight shape: (IN_C, OUT_C, KH, KW)
        # offs_k -> (in_c, kh, kw), offs_n -> out_c
        mask_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Store output
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C) & (batch[:, None] < N)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=mask_out)


def kernel_function(weight, x):
    # weight: (IN_C, OUT_C, KH, KW)
    # x: (N, IN_C, IN_H, IN_W)
    N, IN_C, IN_H, IN_W = x.shape
    IN_C_w, OUT_C, KH, KW = weight.shape

    # stride=1, padding=0, output_padding=0
    OUT_H = (IN_H - 1) * 1 - 2 * 0 + KH
    OUT_W = (IN_W - 1) * 1 - 2 * 0 + KW

    y = torch.zeros(N, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)

    M_TOTAL = N * OUT_H * OUT_W

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv_transpose2d_kernel[grid](
        x, weight, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        M_TOTAL,
    )

    return y


def get_inputs():
    weight = torch.randn(64, 64, 3, 7, dtype=torch.float32, device='cuda')
    x = torch.randn(8, 64, 512, 512, dtype=torch.float32, device='cuda')
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
            print("PASSED")
        else:
            print("FAILED")