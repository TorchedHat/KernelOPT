import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
    ],
    key=['output_length'],
    reset_to_zero=['output_ptr'],
)
@triton.jit
def avg_pool1d_kernel(
    input_ptr, output_ptr,
    batch_size, channels, input_length, output_length,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(0)  # indexes (batch, channel) row
    pid1 = tl.program_id(1)  # indexes output tile within row

    # Scalar CTA-level decomposition — no per-lane vector divisions
    batch = pid0 // channels
    channel = pid0 % channels

    # Per-lane output positions within this row
    out_pos = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_pos < output_length

    # Scalar base offsets for this (batch, channel) row
    row_in_base = batch * channels * input_length + channel * input_length
    row_out_base = batch * channels * output_length + channel * output_length

    win_start = out_pos * stride - padding

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for k in tl.static_range(0, 8):
        in_pos = win_start + k
        valid = (in_pos >= 0) & (in_pos < input_length) & mask
        val = tl.load(input_ptr + row_in_base + in_pos, mask=valid, other=0.0)
        acc += val

    result = acc / kernel_size

    tl.store(output_ptr + row_out_base + out_pos, result, mask=mask)

def kernel_function(x):
    batch_size, channels, input_length = x.shape
    kernel_size = 8
    stride = 1
    padding = 4
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1

    output = torch.empty(batch_size, channels, output_length, device=x.device, dtype=x.dtype)

    total = batch_size * channels * output_length
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)

    avg_pool1d_kernel[grid](
        x, output,
        batch_size, channels, input_length, output_length,
        kernel_size, stride, padding,
    )
    return output


def get_inputs():
    x = torch.rand(64, 128, 65536, device='cuda')
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
        # Verify against PyTorch
        import torch.nn as nn
        avg_pool = nn.AvgPool1d(kernel_size=8, stride=1, padding=4)
        ref = avg_pool(inputs[0])
        print(f"Output shape: {output.shape}, Ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item()}")
        print("Kernel executed successfully.")