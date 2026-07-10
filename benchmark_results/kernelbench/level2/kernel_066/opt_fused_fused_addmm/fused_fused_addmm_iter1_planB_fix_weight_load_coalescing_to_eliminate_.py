import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_softmax_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles one row of output (for softmax)
    # But we need to compute the full row for softmax normalization
    # Strategy: each program_id(0) handles one row, iterates over N in BLOCK_N chunks
    # This is a row-wise softmax after GEMM
    
    row_id = tl.program_id(0)
    
    if row_id >= M:
        return
    
    # We need to compute the full row first, then do softmax
    # For large N (16384), we do online softmax
    
    # Phase 1: Compute GEMM row and find max (online)
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros((), dtype=tl.float32)
    
    # We'll store intermediate results... but 16384 floats per row is too much for SRAM
    # Need a different approach: 2-pass over N
    
    # Actually for softmax we need 2 passes over the data
    # Pass 1: compute all GEMM values and find max + sum
    # Pass 2: normalize
    # But storing 16384 values per row in SRAM is not feasible
    
    # Better: store to global memory, then do softmax
    # But that defeats the purpose of fusion
    
    # Let's do: each row is handled by one program
    # Pass 1: iterate over N in BLOCK_N chunks, compute GEMM tile, track max
    # Pass 2: iterate over N again, compute exp and sum
    # Pass 3: normalize
    # This reads X*K twice but avoids storing intermediate GEMM output
    
    # Actually we need to store the GEMM output temporarily or recompute
    # For 16384 cols, storing in SRAM is not feasible (64KB limit)
    # We'll use global memory as temp buffer
    
    # This approach won't work well as a single kernel without temp storage
    # Let's use a different decomposition
    pass


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit

def gemm_bias_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K
        
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # W is [N, K] (weight matrix), we load W[offs_n, k_offs]
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w = tl.load(w_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)
        
        acc = tl.dot(x, w, acc, allow_tf32=False)
    
    # Add bias
    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    
    # Store output
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


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
def softmax_kernel(
    In_ptr, Out_ptr,
    M, N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= M:
        return
    
    row_start = In_ptr + row_id * stride_m
    
    # Pass 1: find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_start + cols, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Pass 2: compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_start + cols, mask=mask, other=float('-inf'))
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)
    
    # Pass 3: normalize and store
    out_start = Out_ptr + row_id * stride_m
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_start + cols, mask=mask, other=float('-inf'))
        out = tl.exp(x - row_max) / row_sum
        tl.store(out_start + cols, out, mask=mask)


def kernel_function(weight, bias, x):
    # weight: [N, K] = [16384, 16384]
    # bias: [N] = [16384]
    # x: [M, K] = [128, 16384]
    M, K = x.shape
    N = weight.shape[0]
    
    # Allocate intermediate buffer for GEMM output
    gemm_out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    
    # Launch GEMM kernel
    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_kernel[grid_gemm](
        x, weight, bias, gemm_out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        gemm_out.stride(0), gemm_out.stride(1),
    )
    
    # Allocate output for softmax
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    
    # Launch softmax kernel
    grid_softmax = (M,)
    softmax_kernel[grid_softmax](
        gemm_out, out,
        M, N,
        gemm_out.stride(0),
    )
    
    return out


def get_inputs():
    weight = torch.randn(16384, 16384, device='cuda', dtype=torch.float32)
    bias = torch.randn(16384, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 16384, device='cuda', dtype=torch.float32)
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
        # Reference: linear + dropout (no-op at inference) + softmax
        linear_out = x @ weight.T + bias
        ref = torch.softmax(linear_out, dim=1)
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")