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
    # Each program handles one (row_m, n_tile) pair.
    # Computes partial sigmoid sum for BLOCK_N output neurons and stores to Out_ptr (partial buffer).
    row_m  = tl.program_id(0)
    n_tile = tl.program_id(1)
    n_start = n_tile * BLOCK_N
    n_offs  = n_start + tl.arange(0, BLOCK_N)
    n_mask  = n_offs < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k_start in tl.range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        x_ptrs = X_ptr + row_m * stride_xm + k_offs * stride_xk
        x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy='evict_last')
        w_ptrs = W_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk
        w_vals = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0, eviction_policy='evict_first')
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)
    b_vals = tl.load(B_ptr + n_offs, mask=n_mask, other=0.0)
    acc   += b_vals
    sig    = 1.0 / (1.0 + tl.exp(-acc))
    sig    = tl.where(n_mask, sig, 0.0)
    partial_sum = tl.sum(sig, axis=0)
    # Out_ptr here is the partial buffer: shape [M, N_tiles_max], stride_xk used as stride_pn
    tl.store(Out_ptr + row_m * stride_xm + n_tile * stride_xk, partial_sum)


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