import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_sigmoid_sum_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Kernel 1 of 2: partial sigmoid sum for one (row_m, tile_n) pair
    # grid=(M, ceil(N/BLOCK_N)) — each CTA handles exactly one N-tile
    row_m  = tl.program_id(0)
    tile_n = tl.program_id(1)

    n_start = tile_n * BLOCK_N
    n_offs  = n_start + tl.arange(0, BLOCK_N)
    n_mask  = n_offs < N

    # Accumulate dot product for this block of output neurons
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load X[row_m, k_offs]: shape [BLOCK_K]
        x_vals = tl.load(X_ptr + row_m * stride_xm + k_offs * stride_xk,
                         mask=k_mask, other=0.0)

        # Load W[n_offs, k_offs]: shape [BLOCK_N, BLOCK_K]
        w_vals = tl.load(W_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk,
                         mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # acc[n] += sum_k(x[k] * w[n,k])
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Add bias
    b_vals = tl.load(B_ptr + n_offs, mask=n_mask, other=0.0)
    acc += b_vals

    # Apply sigmoid and mask out-of-bounds neurons
    sig = tl.where(n_mask, 1.0 / (1.0 + tl.exp(-acc)), 0.0)

    # Partial sum for this N-tile — store to Out_ptr used as scratch buffer
    # Out_ptr is reinterpreted as (M, num_tiles) scratch; tile_n is the column
    partial_val = tl.sum(sig, axis=0)
    tl.store(Out_ptr + row_m * _PARTIAL_STRIDE_M + tile_n, partial_val)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit

def kernel_function(weight, bias, x):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    
    out = torch.empty(M, 1, device=x.device, dtype=x.dtype)
    
    grid = (M,)
    
    linear_sigmoid_sum_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    
    return out


def get_inputs():
    weight = torch.randn(32768, 32768, device='cuda', dtype=torch.float32)
    bias = torch.randn(32768, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 32768, device='cuda', dtype=torch.float32)
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
        weight, bias, x = inputs
        import torch.nn.functional as F
        linear_out = F.linear(x, weight, bias)
        sig_out = torch.sigmoid(linear_out)
        ref = torch.sum(sig_out, dim=1, keepdim=True)
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")