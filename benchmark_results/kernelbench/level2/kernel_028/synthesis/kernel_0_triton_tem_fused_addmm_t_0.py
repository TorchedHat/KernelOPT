import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    GROUP_M = 8
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (mask_k[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(mask_k[:, None]) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    offs_cm = offs_m
    offs_cn = offs_n
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 4096}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_N': 8192}, num_stages=2, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def instance_norm_add_mul_kernel(
    X_ptr, Y_ptr, Out_ptr,
    M, N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    
    # Compute mean and variance over N elements
    mean = tl.zeros((), dtype=tl.float32)
    var = tl.zeros((), dtype=tl.float32)
    
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    
    mean = mean / N
    
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        var += tl.sum(diff * diff, axis=0)
    
    var = var / N
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(Y_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        x_norm = (x - mean) * inv_std
        result = (x_norm + y) * y
        tl.store(Out_ptr + row * N + cols, result.to(tl.float32), mask=mask)


def kernel_function(p_bmm_weight, p_bmm_bias, x, y):
    M, K = x.shape
    N = p_bmm_weight.shape[0]
    
    # GEMM output
    gemm_out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    
    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_kernel[grid_gemm](
        x, p_bmm_weight, p_bmm_bias, gemm_out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_bmm_weight.stride(1), p_bmm_weight.stride(0),
        gemm_out.stride(0), gemm_out.stride(1),
    )
    
    # Instance norm + add + mul
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid_norm = (M,)
    instance_norm_add_mul_kernel[grid_norm](
        gemm_out, y, out,
        M, N,
        1e-5,
    )
    
    return out


def get_inputs():
    p_bmm_weight = torch.rand(8192, 8192, device='cuda', dtype=torch.float32)
    p_bmm_bias = torch.rand(8192, device='cuda', dtype=torch.float32)
    x = torch.rand(1024, 8192, device='cuda', dtype=torch.float32)
    y = torch.rand(1024, 8192, device='cuda', dtype=torch.float32)
    return [p_bmm_weight, p_bmm_bias, x, y]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        p_bmm_weight, p_bmm_bias, x, y = inputs
        import torch.nn as nn
        linear = nn.Linear(8192, 8192, bias=True).cuda()
        linear.weight.data = p_bmm_weight
        linear.bias.data = p_bmm_bias
        instance_norm = nn.InstanceNorm2d(8192, eps=1e-5).cuda()
        
        with torch.no_grad():
            ref_x = linear(x)
            ref_x = instance_norm(ref_x.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
            ref_x = ref_x + y
            ref_x = ref_x * y
        
        assert torch.allclose(output, ref_x, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref_x).abs().max().item()):.4e}"
        print("Kernel executed successfully.")