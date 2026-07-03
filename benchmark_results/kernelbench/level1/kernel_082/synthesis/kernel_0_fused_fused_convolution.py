import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
    ],
    key=['BATCH', 'OUT_C', 'OUT_H', 'OUT_W', 'IN_H', 'IN_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # X strides: (BATCH, IN_C, IN_H, IN_W) contiguous
    stride_xn: tl.constexpr, stride_xc: tl.constexpr, stride_xh: tl.constexpr, stride_xw: tl.constexpr,
    # W strides: (OUT_C, 1, KERNEL_H, KERNEL_W) - depthwise so group_in_c=1
    stride_wc: tl.constexpr, stride_wkh: tl.constexpr, stride_wkw: tl.constexpr,
    # Y strides: (BATCH, OUT_C, OUT_H, OUT_W) contiguous
    stride_yn: tl.constexpr, stride_yc: tl.constexpr, stride_yh: tl.constexpr, stride_yw: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over M = batch*out_h*out_w
    BLOCK_N: tl.constexpr,  # tile over N = out_channels
):
    # program_id(0): tile over M dimension
    # program_id(1): tile over N dimension (channels)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M_TOTAL
    mask_n = offs_n < OUT_C

    # Decode M -> (batch, out_h, out_w)
    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    batch_idx = tmp // OUT_H

    # Accumulator: [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For depthwise conv: group_in_c = 1, so K = KERNEL_H * KERNEL_W
    # Each output channel c only uses input channel c
    # acc[m, n] = sum_{kh,kw} X[batch[m], n, oh[m]*S+kh-P, ow[m]*S+kw-P] * W[n, 0, kh, kw]

    for kh in tl.static_range(KERNEL_H):
        for kw in tl.static_range(KERNEL_W):
            ih = oh * STRIDE_H + kh - PAD_H  # [BLOCK_M]
            iw = ow * STRIDE_W + kw - PAD_W  # [BLOCK_M]

            valid_h = (ih >= 0) & (ih < IN_H)
            valid_w = (iw >= 0) & (iw < IN_W)
            valid_spatial = valid_h & valid_w  # [BLOCK_M]

            # Load X[batch_idx, offs_n, ih, iw]: shape [BLOCK_M, BLOCK_N]
            # For each (m, n): X[batch_idx[m], offs_n[n], ih[m], iw[m]]
            # x_ptr = X_ptr + batch_idx[m]*stride_xn + offs_n[n]*stride_xc + ih[m]*stride_xh + iw[m]*stride_xw
            x_ptrs = (
            )
            mask_x = mask_m[:, None] & mask_n[None, :] & valid_spatial[:, None]
            x_val = tl.load(x_ptrs, mask=mask_x, other=0.0)  # [BLOCK_M, BLOCK_N]

            # Load W[offs_n, 0, kh, kw]: shape [BLOCK_N]
            # w_ptr = W_ptr + offs_n[n]*stride_wc + kh*stride_wkh + kw*stride_wkw
            w_ptrs = (
            )
            mask_w = mask_n
            w_val = tl.load(w_ptrs, mask=mask_w, other=0.0)  # [BLOCK_N]

            # acc += x_val * w_val[None, :]
            acc += x_val * w_val[None, :]

    # Store output Y[batch_idx, offs_n, oh, ow]
    y_ptrs = (
    )
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=mask_out)


def kernel_function(weight, x):
    # weight: [64, 1, 3, 3], x: [16, 64, 512, 512]
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0

    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

    y = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W

    def grid(meta):
        return (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )

    depthwise_conv2d_kernel[grid](
        x, weight, y,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )
    return y


def get_inputs():
    weight = torch.randn(64, 1, 3, 3, device='cuda', dtype=torch.float32)
    x = torch.randn(16, 64, 512, 512, device='cuda', dtype=torch.float32)
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
        # Verify correctness
        weight, x = inputs
        import torch.nn.functional as F
        ref = F.conv2d(x, weight, bias=None, stride=1, padding=0, groups=64)
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff vs reference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Mismatch: {max_diff}"
        print("Kernel executed successfully.")