import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 2048}, num_warps=8, num_stages=3),
    ],
    key=['output_length', 'channels'],
)
@triton.jit
def avg_pool1d_kernel(
    input_ptr, output_ptr,
    batch_size, channels, input_length, output_length,
    kernel_size, stride, padding,
    XBLOCK: tl.constexpr,
):
    pid_x  = tl.program_id(0)   # tiles along output_length
    pid_bc = tl.program_id(1)   # (batch, channel) pair index
    # Scalar decode done once per CTA, not per thread
    channel = pid_bc % channels
    batch   = pid_bc // channels
    # Thread offsets along output dimension: consecutive threads = consecutive positions
    x_offsets = pid_x * XBLOCK + tl.arange(0, XBLOCK)
    mask = x_offsets < output_length
    win_start = x_offsets * stride - padding
    # Scalar base pointers: no per-thread div/mod
    in_base  = batch * channels * input_length  + channel * input_length
    out_base = batch * channels * output_length + channel * output_length
    acc = tl.zeros([XBLOCK], dtype=tl.float32)
    for k in tl.static_range(0, 8):
        in_pos = win_start + k
        valid  = (in_pos >= 0) & (in_pos < input_length) & mask
        val    = tl.load(input_ptr + in_base + in_pos, mask=valid, other=0.0)
        acc   += val
    result = acc / kernel_size
    tl.store(output_ptr + out_base + x_offsets, result, mask=mask)

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