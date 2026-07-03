import torch
import triton
import triton.language as tl



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
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: axis-0 = (batch, channel) index; axis-1 = out_pos block
    pid_bc  = tl.program_id(0)   # range [0, batch_size * channels)
    pid_pos = tl.program_id(1)   # range [0, cdiv(out_length, BLOCK_SIZE))

    # Derive batch and channel once per CTA (scalar, no per-thread div/mod)
    batch   = pid_bc // channels
    channel = pid_bc %  channels

    # Per-thread offsets along the out_pos dimension
    pos_offsets = pid_pos * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pos_offsets < out_length

    # Input start positions for each output position
    in_start = pos_offsets * stride - padding

    # Base input offset for this (batch, channel) — scalar
    base_offset = batch * (channels * in_length) + channel * in_length

    # Initialize max value to -inf
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    # Unroll over kernel positions
    for k in tl.static_range(kernel_size):
        in_pos = in_start + k * dilation
        valid  = (in_pos >= 0) & (in_pos < in_length) & mask
        val = tl.load(in_ptr + base_offset + in_pos, mask=valid, other=float('-inf'))
        max_val = tl.maximum(max_val, val)

    # Output offset: layout is (batch_size, channels, out_length)
    out_offset = pid_bc * out_length + pos_offsets
    tl.store(out_ptr + out_offset, max_val, mask=mask)

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