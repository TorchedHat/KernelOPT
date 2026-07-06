import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
    ],
    key=['total_elements'],
)
@triton.jit
def max_pool3d_kernel(
    in_ptr,
    out_ptr,
    # Input dimensions
    C, D_in, H_in, W_in,
    # Output dimensions
    D_out, H_out, W_out,
    # Pooling parameters
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    # Strides for input tensor (in elements)
    in_stride_n, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
    # Total output elements
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Decompose flat index into (n, c, od, oh, ow)
    # total_elements = N * C * D_out * H_out * W_out
    # Layout: n, c, od, oh, ow (row-major)
    tmp = offsets
    ow = tmp % W_out
    tmp = tmp // W_out
    oh = tmp % H_out
    tmp = tmp // H_out
    od = tmp % D_out
    tmp = tmp // D_out
    c = tmp % C
    n = tmp // C

    # Compute max over kernel window
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    # Base input pointer offset for (n, c)
    base_nc = n * in_stride_n + c * in_stride_c

    # Starting positions in input (before padding offset)
    d_start = od * stride_d - pad_d
    h_start = oh * stride_h - pad_h
    w_start = ow * stride_w - pad_w

    for kd in tl.static_range(0, 3):  # kernel_d = 3
        id_ = d_start + kd * dil_d
        valid_d = (id_ >= 0) & (id_ < D_in)

        for kh in tl.static_range(0, 3):  # kernel_h = 3
            ih = h_start + kh * dil_h
            valid_h = (ih >= 0) & (ih < H_in)

            for kw in tl.static_range(0, 3):  # kernel_w = 3
                iw = w_start + kw * dil_w
                valid_w = (iw >= 0) & (iw < W_in)

                valid = mask & valid_d & valid_h & valid_w

                in_offset = base_nc + id_ * in_stride_d + ih * in_stride_h + iw * in_stride_w
                val = tl.load(in_ptr + in_offset, mask=valid, other=float('-inf'))
                max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
    ],
    key=['total_elements'],
)
@triton.jit
def max_pool3d_kernel_general(
    in_ptr,
    out_ptr,
    # Input dimensions
    C, D_in, H_in, W_in,
    # Output dimensions
    D_out, H_out, W_out,
    # Pooling parameters
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    # Strides for input tensor (in elements)
    in_stride_n, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
    # Total output elements
    total_elements,
    BLOCK_SIZE: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Decompose flat index into (n, c, od, oh, ow)
    tmp = offsets
    ow = tmp % W_out
    tmp = tmp // W_out
    oh = tmp % H_out
    tmp = tmp // H_out
    od = tmp % D_out
    tmp = tmp // D_out
    c = tmp % C
    n = tmp // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    base_nc = n * in_stride_n + c * in_stride_c

    d_start = od * stride_d - pad_d
    h_start = oh * stride_h - pad_h
    w_start = ow * stride_w - pad_w

    for kd in tl.static_range(0, KD):
        id_ = d_start + kd * dil_d
        valid_d = (id_ >= 0) & (id_ < D_in)

        for kh in tl.static_range(0, KH):
            ih = h_start + kh * dil_h
            valid_h = (ih >= 0) & (ih < H_in)

            for kw in tl.static_range(0, KW):
                iw = w_start + kw * dil_w
                valid_w = (iw >= 0) & (iw < W_in)

                valid = mask & valid_d & valid_h & valid_w

                in_offset = base_nc + id_ * in_stride_d + ih * in_stride_h + iw * in_stride_w
                val = tl.load(in_ptr + in_offset, mask=valid, other=float('-inf'))
                max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


def kernel_function(x):
    # Parameters matching the model
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 3

    N, C, D_in, H_in, W_in = x.shape

    kernel_d = kernel_h = kernel_w = kernel_size
    stride_d = stride_h = stride_w = stride
    pad_d = pad_h = pad_w = padding
    dil_d = dil_h = dil_w = dilation

    # Compute output dimensions (floor mode, no ceil)
    D_out = (D_in + 2 * pad_d - dil_d * (kernel_d - 1) - 1) // stride_d + 1
    H_out = (H_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1

    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    total_elements = N * C * D_out * H_out * W_out

    # Input strides (in elements)
    in_stride_n = x.stride(0)
    in_stride_c = x.stride(1)
    in_stride_d = x.stride(2)
    in_stride_h = x.stride(3)
    in_stride_w = x.stride(4)

    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)

    # Use the specialized kernel for kernel_size=3
    max_pool3d_kernel[grid](
        x, out,
        C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        in_stride_n, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
        total_elements,
    )

    return out


def get_inputs():
    x = torch.rand(16, 32, 128, 128, 128, device='cuda', dtype=torch.float32)
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
        model = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=3)
        ref = model(inputs[0])
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference vs PyTorch: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")
        print(f"Output shape: {output.shape}")