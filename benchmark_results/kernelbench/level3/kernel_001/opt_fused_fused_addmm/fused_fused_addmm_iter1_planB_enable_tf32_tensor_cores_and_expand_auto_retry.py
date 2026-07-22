import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_relu_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    APPLY_RELU: tl.constexpr,
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

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining

        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (mask_k[None, :]), other=0.0, eviction_policy='evict_first')
        b = tl.load(B_ptrs, mask=(mask_k[:, None]) & (offs_n[None, :] < N), other=0.0, eviction_policy='evict_first')

        acc = tl.dot(a, b, acc, allow_tf32=True)

        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Load bias and add
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask)

def kernel_function(w0, b0, w1, b1, w2, b2, x):
    M = x.shape[0]  # 128
    K0 = x.shape[1]  # 16384
    N0 = w0.shape[0]  # 16384
    
    h1 = torch.empty((M, N0), device=x.device, dtype=torch.float32)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N0, meta['BLOCK_N']),)
    gemm_bias_relu_kernel[grid](
        x, w0, b0, h1,
        M, N0, K0,
        x.stride(0), x.stride(1),
        w0.stride(1), w0.stride(0),  # B is w0 transposed: stride_bk=w0.stride(1), stride_bn=w0.stride(0)
        h1.stride(0), h1.stride(1),
        APPLY_RELU=True,
    )
    
    K1 = h1.shape[1]  # 16384
    N1 = w1.shape[0]  # 16384
    h2 = torch.empty((M, N1), device=x.device, dtype=torch.float32)
    
    grid2 = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N1, meta['BLOCK_N']),)
    gemm_bias_relu_kernel[grid2](
        h1, w1, b1, h2,
        M, N1, K1,
        h1.stride(0), h1.stride(1),
        w1.stride(1), w1.stride(0),
        h2.stride(0), h2.stride(1),
        APPLY_RELU=True,
    )
    
    K2 = h2.shape[1]  # 16384
    N2 = w2.shape[0]  # 8192
    out = torch.empty((M, N2), device=x.device, dtype=torch.float32)
    
    grid3 = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N2, meta['BLOCK_N']),)
    gemm_bias_relu_kernel[grid3](
        h2, w2, b2, out,
        M, N2, K2,
        h2.stride(0), h2.stride(1),
        w2.stride(1), w2.stride(0),
        out.stride(0), out.stride(1),
        APPLY_RELU=False,
    )
    
    return out


def get_inputs():
    batch_size = 128
    input_size = 16384
    layer_sizes = [16384, 16384]
    output_size = 8192
    
    w0 = torch.randn(layer_sizes[0], input_size, dtype=torch.float32, device='cuda')
    b0 = torch.randn(layer_sizes[0], dtype=torch.float32, device='cuda')
    w1 = torch.randn(layer_sizes[1], layer_sizes[0], dtype=torch.float32, device='cuda')
    b1 = torch.randn(layer_sizes[1], dtype=torch.float32, device='cuda')
    w2 = torch.randn(output_size, layer_sizes[1], dtype=torch.float32, device='cuda')
    b2 = torch.randn(output_size, dtype=torch.float32, device='cuda')
    x = torch.randn(batch_size, input_size, dtype=torch.float32, device='cuda')
    
    return [w0, b0, w1, b1, w2, b2, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        w0, b0, w1, b1, w2, b2, x = inputs
        import torch.nn.functional as F
        h1_ref = F.relu(F.linear(x, w0, b0))
        h2_ref = F.relu(F.linear(h1_ref, w1, b1))
        ref = F.linear(h2_ref, w2, b2)
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-2), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")