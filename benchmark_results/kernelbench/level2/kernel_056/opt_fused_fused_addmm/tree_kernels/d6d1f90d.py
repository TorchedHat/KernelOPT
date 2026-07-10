import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
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
    # Each program handles BLOCK_M=16 rows of X for better occupancy
    # Computes: sum_n(sigmoid(sum_k(X[m,k] * W[n,k]) + B[n])) for each row m
    BLOCK_M: tl.constexpr = 16
    pid = tl.program_id(0)
    row_m_base = pid * BLOCK_M
    m_offs = row_m_base + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    # Accumulate the final sum over all output neurons, one per row
    total_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    # We tile over the output dimension N
    for n_start in tl.range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # Accumulate dot product for this block of output neurons: [BLOCK_M, BLOCK_N]
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k_start in tl.range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K

            # Load X[m_offs, k_offs]: shape [BLOCK_M, BLOCK_K]
            x_ptrs = X_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
            x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_M, BLOCK_K]

            # Load W[n_offs, k_offs]: shape [BLOCK_N, BLOCK_K]
            w_ptrs = W_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk
            w_vals = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_N, BLOCK_K]

            # acc[m, n] += sum_k(x[m,k] * w[n,k])
            # x_vals: [BLOCK_M, BLOCK_K], tl.trans(w_vals): [BLOCK_K, BLOCK_N]
            # Result: [BLOCK_M, BLOCK_N]
            acc = tl.dot(x_vals, tl.trans(w_vals), acc=acc, allow_tf32=False)

        # Add bias: b_vals [BLOCK_N] -> broadcast to [BLOCK_M, BLOCK_N]
        b_vals = tl.load(B_ptr + n_offs, mask=n_mask, other=0.0)
        acc += b_vals[None, :]

        # Apply sigmoid: 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + tl.exp(-acc))

        # Mask out-of-bounds N and accumulate into total sum per row
        sig = tl.where(n_mask[None, :], sig, 0.0)
        total_sum += tl.sum(sig, axis=1)

    # Store result: coalesced vector store of BLOCK_M consecutive elements
    tl.store(Out_ptr + m_offs, total_sum, mask=m_mask)

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