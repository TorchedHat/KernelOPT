import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def max_pool2d_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    # input dims
    in_H,
    in_W,
    # output dims
    out_H,
    out_W,
    # pooling params
    kernel_size,
    stride,
    padding,
    dilation,
    # strides for input (N, C, H, W)
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # strides for output (N, C, H, W)
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < xnumel

    # Decompose flat index into (n, c, oh, ow) using out strides
    n_idx = offsets // out_stride_n
    rem = offsets % out_stride_n
    c_idx = rem // out_stride_c
    rem2 = rem % out_stride_c
    oh_idx = rem2 // out_stride_h
    ow_idx = rem2 % out_stride_h  # out_stride_h = out_W, so this is ow

    # Compute the starting position in the input
    ih_start = oh_idx * stride - padding
    iw_start = ow_idx * stride - padding

    # Hoist loop-invariant nc_base as scalar before loops
    nc_base = n_idx * in_stride_n + c_idx * in_stride_c

    # Initialize max value to -inf
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    # Iterate over kernel window
    for kh in tl.static_range(0, 4):  # kernel_size=4 hardcoded for autotune
        # Hoist per-kh quantities outside inner kw loop
        ih = ih_start + kh * dilation
        valid_h = (ih >= 0) & (ih < in_H)
        ih_row_offset = nc_base + ih * in_stride_h
        for kw in tl.static_range(0, 4):
            # Only kw-varying quantities inside inner loop
            iw = iw_start + kw * dilation
            valid_w = (iw >= 0) & (iw < in_W)
            valid = valid_h & valid_w & mask
            in_offset = ih_row_offset + iw * in_stride_w
            val = tl.load(in_ptr + in_offset, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    out_offset = n_idx * out_stride_n + c_idx * out_stride_c + oh_idx * out_stride_h + ow_idx * out_stride_w
    tl.store(out_ptr + out_offset, max_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 32},  num_warps=2, num_stages=2),
    ],
    key=['xnumel', 'kernel_size'],
)
@triton.jit

def max_pool2d_kernel_general(
    in_ptr,
    out_ptr,
    xnumel,
    in_H,
    in_W,
    out_H,
    out_W,
    kernel_size,
    stride,
    padding,
    dilation,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < xnumel

    n_idx = offsets // out_stride_n
    rem = offsets % out_stride_n
    c_idx = rem // out_stride_c
    rem2 = rem % out_stride_c
    oh_idx = rem2 // out_stride_h
    ow_idx = rem2 % out_stride_h

    ih_start = oh_idx * stride - padding
    iw_start = ow_idx * stride - padding

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in tl.static_range(0, KERNEL_SIZE):
        for kw in tl.static_range(0, KERNEL_SIZE):
            ih = ih_start + kh * dilation
            iw = iw_start + kw * dilation

            valid_h = (ih >= 0) & (ih < in_H)
            valid_w = (iw >= 0) & (iw < in_W)
            valid = valid_h & valid_w & mask

            in_offset = n_idx * in_stride_n + c_idx * in_stride_c + ih * in_stride_h + iw * in_stride_w
            val = tl.load(in_ptr + in_offset, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    out_offset = n_idx * out_stride_n + c_idx * out_stride_c + oh_idx * out_stride_h + ow_idx * out_stride_w
    tl.store(out_ptr + out_offset, max_val, mask=mask)


def kernel_function(x, kernel_size=4, stride=1, padding=1, dilation=1):
    N, C, H, W = x.shape

    out_H = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_W = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    out = torch.empty((N, C, out_H, out_W), device=x.device, dtype=x.dtype)

    xnumel = N * C * out_H * out_W

    in_stride_n = C * H * W
    in_stride_c = H * W
    in_stride_h = W
    in_stride_w = 1

    out_stride_n = C * out_H * out_W
    out_stride_c = out_H * out_W
    out_stride_h = out_W
    out_stride_w = 1

    grid = lambda meta: (triton.cdiv(xnumel, meta['BLOCK_SIZE']),)

    max_pool2d_kernel_general[grid](
        x,
        out,
        xnumel,
        H, W,
        out_H, out_W,
        kernel_size,
        stride,
        padding,
        dilation,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        KERNEL_SIZE=kernel_size,
    )

    return out


def get_inputs():
    x = torch.rand(32, 64, 512, 512, device='cuda', dtype=torch.float32)
    return [x]


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
        import torch.nn as nn
        ref_model = nn.MaxPool2d(kernel_size=4, stride=1, padding=1, dilation=1).cuda()
        ref_output = ref_model(inputs[0])
        print(f"Output shape: {output.shape}")
        print(f"Reference shape: {ref_output.shape}")
        max_diff = (output - ref_output).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match!")