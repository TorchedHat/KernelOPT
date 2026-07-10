import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 256},  num_warps=4,  num_stages=3, maxnreg=64),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 512},  num_warps=8,  num_stages=3, maxnreg=64),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 512},  num_warps=4,  num_stages=3, maxnreg=64),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 1024}, num_warps=8,  num_stages=2, maxnreg=64),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 1024}, num_warps=4,  num_stages=2, maxnreg=64),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 512},  num_warps=4,  num_stages=3, maxnreg=64),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 1024}, num_warps=8,  num_stages=2, maxnreg=64),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 256},  num_warps=4,  num_stages=3, maxnreg=64),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_sum_logsumexp_kernel(
    X_ptr, W_ptr, B_ptr, OUT_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    = sum_j(sum_k(X[i,k]*W[j,k]) + B[j])
    = sum_k(X[i,k] * sum_j(W[j,k])) + sum_j(B[j])
    
    row_sum[i] = sum_j( (X @ W.T + B)[i,j] )
               = sum_j( sum_k(X[i,k]*W[j,k]) + B[j] )
               = sum_k( X[i,k] * sum_j(W[j,k]) ) + sum_j(B[j])
    
    So we can precompute w_col_sum[k] = sum_j(W[j,k]) and b_sum = sum_j(B[j])
    Then row_sum[i] = dot(X[i,:], w_col_sum) + b_sum
    
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    # Accumulate dot product: X[rows, :] @ w_col_sum[:]
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        k_idx = k_off + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        # Load w_col_sum[k_idx] - precomputed sum over N dimension
        w_col_sum = tl.load(W_ptr + k_idx, mask=k_mask, other=0.0)

        # Load X[rows, k_idx]
        x_ptrs = X_ptr + rows[:, None] * stride_xm + k_idx[None, :] * stride_xk
        x_vals = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate
        acc += tl.sum(x_vals * w_col_sum[None, :], axis=1)

    # Add b_sum (scalar stored at B_ptr[0])
    b_sum = tl.load(B_ptr)
    acc += b_sum

    # Now acc[i] = row_sum[i] for each row
    # After sum: shape [BLOCK_M, 1]
    # max(dim=1, keepdim=True) on [BLOCK_M, 1] -> same value
    # mean(dim=1, keepdim=True) on [BLOCK_M, 1] -> same value
    # logsumexp(dim=1, keepdim=True) on [BLOCK_M, 1] -> same value (logsumexp of 1 element = element)
    # logsumexp(dim=1, keepdim=True) on [BLOCK_M, 1] -> same value
    # So output = acc

    tl.store(OUT_ptr + rows, acc, mask=row_mask)

def kernel_function(p_linear_weight, p_linear_bias, x):
    M, K = x.shape
    N, K2 = p_linear_weight.shape
    assert K == K2

    # Precompute w_col_sum = sum over N (output features) of W[j,k]
    # W shape: [N, K], sum over dim=0 -> [K]
    w_col_sum = p_linear_weight.sum(dim=0)  # [K]
    b_sum_val = p_linear_bias.sum()  # scalar

    # Store b_sum as a 1-element tensor
    b_sum_tensor = b_sum_val.unsqueeze(0)

    out = torch.empty(M, device=x.device, dtype=torch.float32)

    BLOCK_M = 1
    BLOCK_K = 1024
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

    fused_linear_sum_logsumexp_kernel[grid](
        x, w_col_sum, b_sum_tensor, out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_linear_weight.stride(0), p_linear_weight.stride(1),
    )

    return out.view(M, 1)


def get_inputs():
    p_linear_weight = torch.rand(8192, 8192, device='cuda', dtype=torch.float32)
    p_linear_bias = torch.rand(8192, device='cuda', dtype=torch.float32)
    x = torch.rand(1024, 8192, device='cuda', dtype=torch.float32)
    return [p_linear_weight, p_linear_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_linear_weight, p_linear_bias, x = inputs
        # Reference computation
        import torch.nn.functional as F
        linear_out = F.linear(x, p_linear_weight, p_linear_bias)  # [1024, 8192]
        s = torch.sum(linear_out, dim=1, keepdim=True)  # [1024, 1]
        s = torch.max(s, dim=1, keepdim=True)[0]  # [1024, 1]
        s = torch.mean(s, dim=1, keepdim=True)  # [1024, 1]
        s = torch.logsumexp(s, dim=1, keepdim=True)  # [1024, 1]
        ref = torch.logsumexp(s, dim=1, keepdim=True)  # [1024, 1]

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")