import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        # Small tiles, 2 stages: smem=(64+64)*32*2*4=32KB -> 4+ blocks/SM
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        # Medium tiles, 2 stages: smem=(128+64)*32*2*4=49KB -> 4 blocks/SM
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        # Medium tiles, 2-3 stages: smem=(128+128)*32*2*4=64KB -> 3 blocks/SM
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        # Larger M/N tiles, 2 stages: smem=(256+128)*32*2*4=96KB -> 2 blocks/SM
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        # Wide tile for N-heavy workload (N=8192)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=8),
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
    
    # Create block pointers for X and W using TMA-compatible make_block_ptr
    # X has shape (M, K) with strides (stride_xm, stride_xk)
    # order=(1, 0) means K-dimension (dim 1) is the fastest-varying in the block
    X_block_ptr = tl.make_block_ptr(
        base=X_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # W is stored as (N, K) but we treat it as (K, N) for X @ W^T
    # shape=(K, N), strides=(stride_wk, stride_wn) = (1, K)
    # order=(1, 0) means N-dimension (dim 1) is the fastest-varying in the block,
    # matching TMA swizzle layout and eliminating shared memory bank conflicts
    W_block_ptr = tl.make_block_ptr(
        base=W_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    
    # Accumulate in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load X tile [BLOCK_M, BLOCK_K] using block pointer (TMA swizzled layout)
        x = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option='zero')
        
        # Load W tile [BLOCK_K, BLOCK_N] using block pointer (TMA swizzled layout)
        w = tl.load(W_block_ptr, boundary_check=(0, 1), padding_option='zero')
        
        # tf32: single-pass HMMA — lower register pressure, enables 4 blocks/SM
        acc = tl.dot(x, w, acc, input_precision='tf32')
        
        # Advance block pointers to next K tile
        X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_K))
        W_block_ptr = tl.advance(W_block_ptr, (BLOCK_K, 0))
    
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