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
    key=['batch_size', 'channels', 'output_length'],
)
@triton.jit
def avg_pool1d_kernel(
    input_ptr, output_ptr,
    batch_size, channels, input_length, output_length,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = batch_size * channels * output_length
    mask = offsets < total

    out_pos = offsets % output_length
    tmp = offsets // output_length
    channel = tmp % channels
    batch = tmp // channels

    win_start = out_pos * stride - padding

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    inv_kernel_size = 1.0 / kernel_size
    for k in tl.static_range(0, 8):
        in_pos = win_start + k
        valid = (in_pos >= 0) & (in_pos < input_length) & mask
        in_idx = batch * channels * input_length + channel * input_length + in_pos
        val = tl.load(input_ptr + in_idx, mask=valid, other=0.0)
        acc += val * inv_kernel_size

    out_idx = batch * channels * output_length + channel * output_length + out_pos
    tl.store(output_ptr + out_idx, acc, mask=mask)

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