import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 8192}, num_warps=16, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_mean_var_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per batch element
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N

    # Welford online algorithm for mean and variance
    mean = tl.zeros([], dtype=tl.float32)
    m2 = tl.zeros([], dtype=tl.float32)
    count = tl.zeros([], dtype=tl.float32)

    for off in tl.range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        # Welford update
        count_new = count + tl.sum(mask.to(tl.float32), axis=0)
        delta = x - mean
        # Only update for valid elements
        x_masked = tl.where(mask, x, mean)
        # Process block with Welford
        block_count = tl.sum(mask.to(tl.float32), axis=0)
        block_mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / tl.maximum(block_count, 1.0)
        block_m2 = tl.sum(tl.where(mask, (x - block_mean) * (x - block_mean), 0.0), axis=0)
        
        # Merge
        new_count = count + block_count
        delta2 = block_mean - mean
        new_mean = tl.where(new_count > 0, mean + delta2 * block_count / new_count, mean)
        new_m2 = m2 + block_m2 + delta2 * delta2 * count * block_count / tl.maximum(new_count, 1.0)
        
        mean = tl.where(new_count > 0, new_mean, mean)
        m2 = tl.where(new_count > 0, new_m2, m2)
        count = new_count

    variance = m2 / count
    tl.store(mean_ptr + row, mean)
    tl.store(var_ptr + row, variance)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 8192}, num_warps=16, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_normalize_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    out_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    mean = tl.load(mean_ptr + row).to(tl.float32)
    var = tl.load(var_ptr + row).to(tl.float32)
    rstd = libdevice.rsqrt(var + eps)

    for off in tl.range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_norm = (x - mean) * rstd
        out = x_norm * w + b
        tl.store(out_row_ptr + cols, out.to(tl.float16 if out_ptr.dtype.element_ty == tl.float16 else tl.float32), mask=mask)


def kernel_function(weight, bias, x):
    # x: [16, 64, 256, 256]
    # weight: [64, 256, 256]
    # bias: [64, 256, 256]
    B = x.shape[0]
    N = x.shape[1] * x.shape[2] * x.shape[3]  # 4194304
    
    x_flat = x.reshape(B, N)
    weight_flat = weight.reshape(N)
    bias_flat = bias.reshape(N)
    
    mean = torch.empty(B, device=x.device, dtype=torch.float32)
    var = torch.empty(B, device=x.device, dtype=torch.float32)
    out = torch.empty_like(x)
    out_flat = out.reshape(B, N)
    
    grid_mv = (B,)
    layer_norm_mean_var_kernel[grid_mv](
    )
    
    grid_norm = (B,)
    layer_norm_normalize_kernel[grid_norm](
    )
    
    return out


def get_inputs():
    weight = torch.rand(64, 256, 256, device='cuda', dtype=torch.float32)
    bias = torch.rand(64, 256, 256, device='cuda', dtype=torch.float32)
    x = torch.rand(16, 64, 256, 256, device='cuda', dtype=torch.float32)
    return [weight, bias, x]


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
        weight, bias, x = inputs
        ln = nn.LayerNorm(normalized_shape=(64, 256, 256)).cuda()
        ln.weight.data = weight
        ln.bias.data = bias
        ref = ln(x)
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")