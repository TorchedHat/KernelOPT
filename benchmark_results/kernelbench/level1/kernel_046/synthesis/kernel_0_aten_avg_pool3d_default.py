import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_total'],
)
@triton.jit
def avg_pool3d_kernel(
    input_ptr,
    output_ptr,
    # input shape
    in_B, in_C, in_D, in_H, in_W,
    # output shape
    out_D, out_H, out_W,
    # strides for input (in elements)
    in_stride_b, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
    # strides for output (in elements)
    out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    # pooling params
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    # total output elements
    N_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_total

    # Decompose flat index into (b, c, od, oh, ow)
    tmp = offsets
    ow = tmp % out_W
    tmp = tmp // out_W
    oh = tmp % out_H
    tmp = tmp // out_H
    od = tmp % out_D
    tmp = tmp // out_D
    c = tmp % in_C
    b = tmp // in_C

    # Compute input start positions
    id_start = od * stride_d - pad_d
    ih_start = oh * stride_h - pad_h
    iw_start = ow * stride_w - pad_w

    # Accumulate sum over kernel window
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for kd in range(kernel_d):
        id_cur = id_start + kd
        valid_d = (id_cur >= 0) & (id_cur < in_D)
        for kh in range(kernel_h):
            ih_cur = ih_start + kh
            valid_h = (ih_cur >= 0) & (ih_cur < in_H)
            for kw in range(kernel_w):
                iw_cur = iw_start + kw
                valid_w = (iw_cur >= 0) & (iw_cur < in_W)
                valid = mask & valid_d & valid_h & valid_w

                # Compute linear index into input
                in_idx = (b * in_stride_b +

                val = tl.load(input_ptr + in_idx, mask=valid, other=0.0)
                acc = acc + val
                count = count + tl.where(valid_d & valid_h & valid_w & mask, 1.0, 0.0)

    # count_inv: use count_inv = 1.0 / (kernel_d * kernel_h * kernel_w) for count_include_pad=True
    # The PyTorch model uses count_include_pad=True (default for AvgPool3d with padding)
    # Actually nn.AvgPool3d default: count_include_pad=True
    kernel_vol = kernel_d * kernel_h * kernel_w
    result = acc / kernel_vol

    # Compute output index
    out_idx = (b * out_stride_b +

    tl.store(output_ptr + out_idx, result.to(tl.float32), mask=mask)


def kernel_function(x):
    # x: (B, C, D, H, W)
    B, C, D, H, W = x.shape
    kernel_size = 3
    stride = 2
    padding = 1

    out_D = (D + 2 * padding - kernel_size) // stride + 1
    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1

    output = torch.empty((B, C, out_D, out_H, out_W), device=x.device, dtype=x.dtype)

    N_total = B * C * out_D * out_H * out_W

    # Input strides
    in_stride_b = x.stride(0)
    in_stride_c = x.stride(1)
    in_stride_d = x.stride(2)
    in_stride_h = x.stride(3)
    in_stride_w = x.stride(4)

    # Output strides
    out_stride_b = output.stride(0)
    out_stride_c = output.stride(1)
    out_stride_d = output.stride(2)
    out_stride_h = output.stride(3)
    out_stride_w = output.stride(4)

    grid = lambda meta: (triton.cdiv(N_total, meta['BLOCK_SIZE']),)

    avg_pool3d_kernel[grid](
        x, output,
        B, C, D, H, W,
        out_D, out_H, out_W,
        in_stride_b, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
        kernel_size, kernel_size, kernel_size,
        stride, stride, stride,
        padding, padding, padding,
        N_total,
    )

    return output


def get_inputs():
    x = torch.rand(16, 32, 128, 128, 256, device='cuda', dtype=torch.float32)
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
        ref_pool = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        ref_out = ref_pool(inputs[0])
        max_diff = (output - ref_out).abs().max().item()
        print(f"Max difference from reference: {max_diff}")
        assert max_diff < 1e-3, f"Correctness check failed: max_diff={max_diff}"
        print("Kernel executed successfully and matches reference.")