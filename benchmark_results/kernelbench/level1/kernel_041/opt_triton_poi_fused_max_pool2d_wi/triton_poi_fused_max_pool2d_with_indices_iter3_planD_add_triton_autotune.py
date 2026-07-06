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
    key=['total_elements'],
)
@triton.jit
def max_pool1d_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    channels,
    in_length,
    out_length,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.multiple_of(pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    mask = offsets < total_elements

    # Decompose flat index into (batch, channel, out_pos)
    out_pos = offsets % out_length
    tmp = offsets // out_length
    channel = tmp % channels
    batch = tmp // channels

    # Compute the base input position
    in_start = out_pos * stride - padding

    # Compute the channel stride and batch stride
    ch_stride = in_length
    batch_stride = channels * in_length

    # Base pointer offset for this (batch, channel)
    base_offset = batch * batch_stride + channel * ch_stride

    # Initialize max value to -inf
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    # Unroll over kernel positions
    for k in tl.static_range(kernel_size):
        in_pos = in_start + k * dilation
        valid = (in_pos >= 0) & (in_pos < in_length) & mask
        load_offset = tl.multiple_of(base_offset + in_pos, 1)
        val = tl.load(in_ptr + load_offset, mask=valid, other=float('-inf'))
        max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offsets, max_val, mask=mask)

def kernel_function(x):
    # Parameters matching the model
    kernel_size = 8
    stride = 1
    padding = 4
    dilation = 3

    batch_size, channels, in_length = x.shape

    # Compute output length
    out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    out = torch.empty((batch_size, channels, out_length), dtype=x.dtype, device=x.device)

    total_elements = batch_size * channels * out_length

    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)

    max_pool1d_kernel[grid](
        x,
        out,
        batch_size,
        channels,
        in_length,
        out_length,
        kernel_size,
        stride,
        padding,
        dilation,
        total_elements,
    )

    return out


def get_inputs():
    x = torch.rand(64, 192, 65536, device='cuda', dtype=torch.float32)
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
        # Verify correctness against PyTorch
        import torch.nn as nn
        maxpool = nn.MaxPool1d(kernel_size=8, stride=1, padding=4, dilation=3).cuda()
        ref_output = maxpool(inputs[0])
        print(f"Output shape: {output.shape}")
        print(f"Reference shape: {ref_output.shape}")
        max_diff = (output - ref_output).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match!")