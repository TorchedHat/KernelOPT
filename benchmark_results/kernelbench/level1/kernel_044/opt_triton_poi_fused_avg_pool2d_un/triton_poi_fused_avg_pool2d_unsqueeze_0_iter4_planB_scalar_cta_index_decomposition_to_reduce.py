import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=1),
    ],
    key=['output_length'],
)
@triton.jit
def avg_pool1d_kernel(
    input_ptr, output_ptr,
    batch_size, channels, input_length, output_length,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total = batch_size * channels * output_length
    num_tiles = tl.cdiv(total, BLOCK_SIZE)

    while tile_id < num_tiles:
        # Scalar CTA-level decomposition: replace 4 vector div/mod chains with
        # 3 scalar integer ops + 1 vector mod for boundary handling
        tile_start = tile_id * BLOCK_SIZE          # scalar
        out_start_s = tile_start % output_length   # scalar
        tmp_s = tile_start // output_length        # scalar
        channel_s = tmp_s % channels               # scalar
        batch_s = tmp_s // channels                # scalar

        local_off = tl.arange(0, BLOCK_SIZE)       # [0..BLOCK_SIZE-1]
        global_off = tile_start + local_off        # for mask only
        mask = global_off < total

        # out_pos as simple vector add (no div), then handle row-boundary wrap
        out_pos = out_start_s + local_off          # vector add (no div)
        row_inc = out_pos // output_length         # 0 or 1 per thread (boundary)
        out_pos = out_pos % output_length          # 1 vector mod (was 4 vector div/mod)
        channel_v = channel_s + row_inc            # vector add
        batch_v = batch_s + (channel_v // channels)  # vector add
        channel_v = channel_v % channels

        win_start = out_pos * stride - padding

        inv_ks = 1.0 / kernel_size  # Python float = 0.125, exact in float32, folded as constexpr
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        # FIX: inline address — no hoisted base_in/base_out
        for k in tl.static_range(0, 8):
            in_pos = win_start + k
            valid  = (in_pos >= 0) & (in_pos < input_length) & mask
            in_idx = batch_v * (channels * input_length) + channel_v * input_length + in_pos
            val    = tl.load(input_ptr + in_idx, mask=valid, other=0.0)
            acc    = tl.math.fma(val, inv_ks, acc)  # FFMA: acc = val*inv_ks + acc

        result = acc  # division already absorbed — no post-loop op needed
        out_idx = batch_v * (channels * output_length) + channel_v * output_length + out_pos
        tl.store(output_ptr + out_idx, result, mask=mask)
        tile_id += num_programs

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