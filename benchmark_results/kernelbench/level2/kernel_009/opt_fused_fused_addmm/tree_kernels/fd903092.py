import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_sub_mul_relu_kernel(
    # Pointers
    X_ptr, W_ptr, B_ptr, Out_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    # Scalar constants
    subtract_value, multiply_value,
    # Tile sizes (compile-time constants)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Grid dimensions
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 cache-friendly reordering
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    
    # Row/col offsets for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    # Boundary-safe offsets
    offs_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    
    # Pointers to first tiles
    X_ptrs = X_ptr + offs_m[:, None] * stride_xm + rk[None, :] * stride_xk
    # W is stored as (N, K) i.e. weight matrix transposed from (K, N)
    W_ptrs = W_ptr + offs_n[None, :] * stride_wn + rk[:, None] * stride_wk
    
    # Accumulate in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        # Load X tile [BLOCK_M, BLOCK_K]
        x_mask = (offs_m[:, None] < M) & (rk[None, :] < k_remaining)
        x = tl.load(X_ptrs, mask=x_mask, other=0.0, eviction_policy='evict_first')
        
        # Load W tile [BLOCK_K, BLOCK_N]
        w_mask = (rk[:, None] < k_remaining) & (offs_n[None, :] < N)
        w = tl.load(W_ptrs, mask=w_mask, other=0.0, eviction_policy='evict_first')
        
        acc = tl.dot(x, w, acc, allow_tf32=False)
        
        X_ptrs += BLOCK_K * stride_xk
        W_ptrs += BLOCK_K * stride_wk
    
    # Recompute rm, rn for output
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Load bias [BLOCK_N]
    bias_mask = rn < N
    bias = tl.load(B_ptr + rn, mask=bias_mask, other=0.0)
    
    # Fused epilogue: add bias, subtract, multiply, relu
    result = acc + bias[None, :]
    result = result - subtract_value
    result = result * multiply_value
    result = tl.maximum(result, 0.0)
    
    # Store output
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    Out_ptrs = Out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(Out_ptrs, result.to(tl.float32), mask=out_mask)

def kernel_function(p_linear_weight, p_linear_bias, x):
    # x: [M, K], weight: [N, K] (linear weight is stored as [out, in])
    M, K = x.shape
    N = p_linear_weight.shape[0]
    
    # Allocate output
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Weight is [N, K], we want X @ W^T = X @ weight^T
    # So we treat weight as (N, K) with strides (K, 1)
    # and compute X[M,K] @ W^T[K,N] = out[M,N]
    # In the kernel: W_ptr indexed as offs_n * stride_wn + rk * stride_wk
    # where stride_wn = K (row stride of weight), stride_wk = 1 (col stride)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    fused_linear_sub_mul_relu_kernel[grid](
        x, p_linear_weight, p_linear_bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_linear_weight.stride(0), p_linear_weight.stride(1),
        out.stride(0), out.stride(1),
        2.0, 1.5,
    )
    
    return out


def get_inputs():
    p_linear_weight = torch.randn(8192, 8192, dtype=torch.float32, device='cuda')
    p_linear_bias = torch.randn(8192, dtype=torch.float32, device='cuda')
    x = torch.rand(1024, 8192, dtype=torch.float32, device='cuda')
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
        # Verify correctness against PyTorch reference
        p_linear_weight, p_linear_bias, x = inputs
        ref = torch.nn.functional.linear(x, p_linear_weight, p_linear_bias)
        ref = ref - 2.0
        ref = ref * 1.5
        ref = torch.relu(ref)
        
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference from reference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match reference.")