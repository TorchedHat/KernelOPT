import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_R': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_R': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_R': 512},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 512},  num_warps=4, num_stages=4),
    ],
    key=['C_per_group', 'HW'],
)
@triton.jit
def group_norm_kernel(
    X_ptr,
    W_ptr,
    B_ptr,
    Y_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    G: tl.constexpr,
    C_per_group: tl.constexpr,
    eps,
    BLOCK_R: tl.constexpr,
):
    # Each program handles one (batch_idx, group_idx) pair
    pid = tl.program_id(0)
    batch_idx = pid // G
    group_idx = pid % G

    # Base pointer for this (batch, group)
    # X shape: [N, C, HW] logically
    # group starts at channel: group_idx * C_per_group
    base_offset = batch_idx * C * HW + group_idx * C_per_group * HW

    # Pass 1: Compute mean and variance using Welford's online algorithm
    # Restructured to channel-outer, spatial-inner for cache locality
    mean = tl.zeros((), dtype=tl.float32)
    m2 = tl.zeros((), dtype=tl.float32)
    count = tl.zeros((), dtype=tl.float32)

    for c in tl.static_range(0, C_per_group):
        base_offset_c = base_offset + c * HW
        for hw_start in tl.range(0, HW, BLOCK_R):
            hw_offsets = hw_start + tl.arange(0, BLOCK_R)
            mask = hw_offsets < HW
            x = tl.load(X_ptr + base_offset_c + hw_offsets, mask=mask, other=0.0).to(tl.float32)

            # Welford update
            block_count = tl.sum(mask.to(tl.float32), axis=0)
            block_mean = tl.sum(tl.where(mask, x, tl.zeros_like(x)), axis=0) / tl.maximum(block_count, 1.0)
            block_m2 = tl.sum(tl.where(mask, (x - block_mean) * (x - block_mean), tl.zeros_like(x)), axis=0)

            # Merge with running stats
            new_count = count + block_count
            delta = block_mean - mean
            mean = tl.where(new_count > 0, mean + delta * block_count / new_count, mean)
            m2 = m2 + block_m2 + delta * delta * count * block_count / tl.maximum(new_count, 1.0)
            count = new_count

    variance = m2 / tl.maximum(count, 1.0)
    inv_std = libdevice.rsqrt(variance + eps)

    # Pass 2: Normalize and apply weight/bias
    # channel-outer, spatial-inner: load w and b once per channel as scalars
    for c in tl.static_range(0, C_per_group):
        channel_idx = group_idx * C_per_group + c
        w = tl.load(W_ptr + channel_idx).to(tl.float32)   # scalar, loaded once per channel
        b = tl.load(B_ptr + channel_idx).to(tl.float32)   # scalar, loaded once per channel
        base_offset_c = base_offset + c * HW
        for hw_start in tl.range(0, HW, BLOCK_R):
            hw_offsets = hw_start + tl.arange(0, BLOCK_R)
            mask = hw_offsets < HW
            x = tl.load(X_ptr + base_offset_c + hw_offsets, mask=mask, other=0.0).to(tl.float32)
            x_norm = (x - mean) * inv_std
            y = x_norm * w + b
            tl.store(Y_ptr + base_offset_c + hw_offsets, y.to(tl.float16 if Y_ptr.dtype.element_ty == tl.float16 else tl.float32), mask=mask)

def kernel_function(weight, bias, x):
    N, C, H, W = x.shape
    G = 8  # num_groups
    C_per_group = C // G
    HW = H * W
    eps = 1e-5

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    # One program per (batch, group) pair
    total_programs = N * G

    grid = (total_programs,)

    group_norm_kernel[grid](
        x_contig,
        weight,
        bias,
        y,
        N, C, HW, G, C_per_group,
        eps,
    )

    return y


def get_inputs():
    x = torch.rand(112, 64, 512, 512, device='cuda', dtype=torch.float32)
    weight = torch.rand(64, device='cuda', dtype=torch.float32)
    bias = torch.rand(64, device='cuda', dtype=torch.float32)
    return [weight, bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    weight, bias, x = inputs

    # Verify correctness
    if not args.profile_only:
        import torch.nn as nn
        gn = nn.GroupNorm(num_groups=8, num_channels=64).cuda()
        with torch.no_grad():
            gn.weight.copy_(weight)
            gn.bias.copy_(bias)
            ref = gn(x)

        out = kernel_function(weight, bias, x)
        max_diff = (out - ref).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")