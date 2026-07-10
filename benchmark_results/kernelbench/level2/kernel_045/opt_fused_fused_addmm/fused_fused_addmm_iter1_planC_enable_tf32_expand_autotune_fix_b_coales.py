import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_sigmoid_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    APPLY_SIGMOID: tl.constexpr,
    ADD_BIAS: tl.constexpr,
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
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy='evict_first')
        
        # B is weight matrix (N, K), we access B[n, k] = B_ptr + n*stride_bn + k*stride_bk
        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + k_offs[:, None] * stride_bk
        b_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0, eviction_policy='evict_first')
        
        acc = tl.dot(a, b, acc, allow_tf32=True)
    
    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)
    
    if ADD_BIAS:
        bias = tl.load(bias_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
        acc = acc + bias[None, :]
    
    if APPLY_SIGMOID:
        acc = tl.sigmoid(acc)
    
    c_ptrs = C_ptr + offs_m_out[:, None] * stride_cm + offs_n_out[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 128}, num_stages=2, num_warps=4),
    ],
    key=['N'],
)
@triton.jit

def logsumexp_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(input_ptr + row_idx * stride_m + cols * stride_n, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Handle inf max
    abs_max = tl.abs(row_max)
    row_max = tl.where(abs_max == float('inf'), 0.0, row_max)
    
    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(input_ptr + row_idx * stride_m + cols * stride_n, mask=mask, other=float('-inf'))
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)
    
    result = tl.log(row_sum) + row_max
    tl.store(output_ptr + row_idx, result)


def kernel_function(w1, b1, w2, b2, x):
    M = x.shape[0]
    K1 = x.shape[1]
    N1 = w1.shape[0]  # hidden_size = 4096
    N2 = w2.shape[0]  # output_size = 1024
    
    # First GEMM: x @ w1^T + b1, then sigmoid
    hidden = torch.empty((M, N1), device=x.device, dtype=torch.float32)
    
    grid1 = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N1, meta['BLOCK_N']),)
    gemm_sigmoid_kernel[grid1](
        x, w1, hidden, b1,
        M, N1, K1,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        hidden.stride(0), hidden.stride(1),
        APPLY_SIGMOID=True,
        ADD_BIAS=True,
    )
    
    # Second GEMM: hidden @ w2^T + b2
    out2 = torch.empty((M, N2), device=x.device, dtype=torch.float32)
    
    grid2 = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N2, meta['BLOCK_N']),)
    gemm_sigmoid_kernel[grid2](
        hidden, w2, out2, b2,
        M, N2, N1,
        hidden.stride(0), hidden.stride(1),
        w2.stride(0), w2.stride(1),
        out2.stride(0), out2.stride(1),
        APPLY_SIGMOID=False,
        ADD_BIAS=True,
    )
    
    # LogSumExp over dim=1
    output = torch.empty((M,), device=x.device, dtype=torch.float32)
    
    grid3 = (M,)
    logsumexp_kernel[grid3](
        out2, output,
        M, N2,
        out2.stride(0), out2.stride(1),
    )
    
    return output


def get_inputs():
    w1 = torch.randn(4096, 2048, dtype=torch.float32, device='cuda')
    b1 = torch.randn(4096, dtype=torch.float32, device='cuda')
    w2 = torch.randn(1024, 4096, dtype=torch.float32, device='cuda')
    b2 = torch.randn(1024, dtype=torch.float32, device='cuda')
    x = torch.rand(16384, 2048, dtype=torch.float32, device='cuda')
    return [w1, b1, w2, b2, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    w1, b1, w2, b2, x = inputs
    
    output = kernel_function(w1, b1, w2, b2, x)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        import torch.nn.functional as F
        h = F.linear(x, w1, b1)
        h = torch.sigmoid(h)
        h = F.linear(h, w2, b2)
        ref = torch.logsumexp(h, dim=1)
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")