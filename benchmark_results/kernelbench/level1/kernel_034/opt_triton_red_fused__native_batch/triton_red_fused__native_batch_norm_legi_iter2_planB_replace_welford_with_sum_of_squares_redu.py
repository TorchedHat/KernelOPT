import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def instance_norm_kernel(
    x_ptr, out_ptr,
    N,  # number of spatial elements per (batch, channel) = H*W
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    # Pass 1: sum + sum-of-squares for mean and variance
    x_sum    = tl.zeros((), dtype=tl.float32)
    x_sq_sum = tl.zeros((), dtype=tl.float32)

    for off in tl.range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
        x_sum    += tl.sum(tl.where(mask, x,     tl.zeros_like(x)), axis=0)
        x_sq_sum += tl.sum(tl.where(mask, x * x, tl.zeros_like(x)), axis=0)

    N_f     = N.to(tl.float32)
    mean    = x_sum / N_f
    var     = x_sq_sum / N_f - mean * mean
    inv_std = libdevice.rsqrt(var + eps)

    # Pass 2: Normalize
    for off in tl.range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
        out = (x - mean) * inv_std
        tl.store(out_ptr + row_start + cols, out.to(tl.float32), mask=mask, eviction_policy='evict_first')

def kernel_function(*inputs):
    x, = inputs
    assert x.is_cuda and x.dtype == torch.float32
    
    B, C, H, W = x.shape
    N = H * W  # spatial elements per instance
    num_rows = B * C  # number of (batch, channel) pairs
    
    # Flatten to [B*C, H*W] view
    x_flat = x.reshape(num_rows, N)
    out_flat = torch.empty_like(x_flat)
    
    eps = 1e-5
    grid = (num_rows,)
    
    instance_norm_kernel[grid](
        x_flat, out_flat,
        N,
        eps,
    )
    
    return out_flat.reshape(B, C, H, W)


def get_inputs():
    x = torch.rand(112, 64, 512, 512, device='cuda', dtype=torch.float32)
    return [x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    # Verify correctness
    if not args.profile_only:
        import torch.nn as nn
        model = nn.InstanceNorm2d(num_features=64).cuda()
        model.eval()
        with torch.no_grad():
            ref = model(inputs[0])
        out = kernel_function(*inputs)
        max_diff = (out - ref).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Correctness check failed: max_diff={max_diff}"
        print("Correctness check passed!")
    
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        print("Kernel executed successfully.")