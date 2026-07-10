import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# GEMM kernel: computes x @ W.T (W is [N,K], so W.T is [K,N])
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
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
    
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
    
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


# Kernel 2: Fused swish + bias + group_norm mean/var computation
# Each program handles one (batch_row, group) pair
# x shape: [M, N] where M=32768, N=4096, num_groups=64, group_size=64
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
    ],
    key=['GROUP_SIZE'],
)
@triton.jit
def swish_bias_groupnorm_stats_kernel(
    M, N,
    num_groups,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (row, group) pair
    pid = tl.program_id(0)
    row = pid // num_groups
    group = pid % num_groups
    
    if row >= M:
        return
    
    group_start = group * GROUP_SIZE
    
    # Compute mean first
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, GROUP_SIZE, BLOCK_SIZE):
        cols = group_start + off + tl.arange(0, BLOCK_SIZE)
        mask = (cols < group_start + GROUP_SIZE) & (cols < N)
        x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(BIAS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        # Apply swish: sigmoid(x) * x
        sig = 1.0 / (1.0 + tl.exp(-x))
        x_swish = sig * x
        x_biased = x_swish + bias
        mean += tl.sum(x_biased, axis=0)
    mean = mean / GROUP_SIZE
    
    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, GROUP_SIZE, BLOCK_SIZE):
        cols = group_start + off + tl.arange(0, BLOCK_SIZE)
        mask = (cols < group_start + GROUP_SIZE) & (cols < N)
        x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(BIAS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-x))
        x_swish = sig * x
        x_biased = x_swish + bias
        diff = tl.where(mask, x_biased - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    
    tl.store(MEAN_ptr + row * num_groups + group, mean)
    tl.store(VAR_ptr + row * num_groups + group, var)


# Kernel 3: Final normalization pointwise
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=2, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def groupnorm_normalize_kernel(
    M, N,
    num_groups,
    GROUP_SIZE,
    eps,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    
    row = offs // N
    col = offs % N
    group = col // GROUP_SIZE
    
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(BIAS_ptr + col, mask=mask, other=0.0).to(tl.float32)
    
    # Apply swish
    sig = 1.0 / (1.0 + tl.exp(-x))
    x_swish = sig * x
    x_biased = x_swish + bias
    
    mean = tl.load(MEAN_ptr + row * num_groups + group, mask=mask, other=0.0).to(tl.float32)
    var_sum = tl.load(VAR_ptr + row * num_groups + group, mask=mask, other=0.0).to(tl.float32)
    
    var = var_sum / GROUP_SIZE
    rstd = libdevice.rsqrt(var + eps)
    
    x_norm = (x_biased - mean) * rstd
    
    gn_w = tl.load(GN_W_ptr + col, mask=mask, other=1.0).to(tl.float32)
    gn_b = tl.load(GN_B_ptr + col, mask=mask, other=0.0).to(tl.float32)
    
    out = x_norm * gn_w + gn_b
    tl.store(OUT_ptr + offs, out, mask=mask)


def kernel_function(p_matmul_weight, p_matmul_bias, p_bias, p_group_norm_weight, p_group_norm_bias, x):
    # x: [32768, 1024]
    # p_matmul_weight: [4096, 1024]
    # p_matmul_bias: [4096]
    # p_bias: [4096]
    # p_group_norm_weight: [4096]
    # p_group_norm_bias: [4096]
    
    M, K = x.shape
    N = p_matmul_weight.shape[0]  # 4096
    
    # Step 1: GEMM x @ W.T + linear_bias
    # W is [N, K], W.T is [K, N]
    # A = x: [M, K], stride_am=K, stride_ak=1
    # B = W: [N, K], we access as W.T so stride_bk=1, stride_bn=K
    mm_out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    
    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_kernel[grid_gemm](
        x, p_matmul_weight, mm_out, p_matmul_bias,
        M, N, K,
        x.stride(0), x.stride(1),           # stride_am, stride_ak
        p_matmul_weight.stride(1), p_matmul_weight.stride(0),  # stride_bk=stride(1), stride_bn=stride(0) for W.T
        mm_out.stride(0), mm_out.stride(1),
        HAS_BIAS=True,
    )
    
    # Step 2: Compute group norm stats (after swish + bias)
    num_groups = 64
    group_size = N // num_groups  # 64
    
    mean_buf = torch.empty((M, num_groups), device=x.device, dtype=torch.float32)
    var_buf = torch.empty((M, num_groups), device=x.device, dtype=torch.float32)
    
    grid_stats = (M * num_groups,)
    swish_bias_groupnorm_stats_kernel[grid_stats](
        mm_out, p_bias, mean_buf, var_buf,
        M, N, num_groups, group_size,
    )
    
    # Step 3: Final normalization
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    total = M * N
    grid_norm = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    groupnorm_normalize_kernel[grid_norm](
        mm_out, p_bias, mean_buf, var_buf,
        p_group_norm_weight, p_group_norm_bias,
        out,
        M, N, num_groups, group_size,
        1e-5, total,
    )
    
    return out


def get_inputs():
    torch.manual_seed(42)
    p_matmul_weight = torch.randn(4096, 1024, device='cuda', dtype=torch.float32)
    p_matmul_bias = torch.randn(4096, device='cuda', dtype=torch.float32)
    p_bias = torch.randn(4096, device='cuda', dtype=torch.float32)
    p_group_norm_weight = torch.randn(4096, device='cuda', dtype=torch.float32)
    p_group_norm_bias = torch.randn(4096, device='cuda', dtype=torch.float32)
    x = torch.rand(32768, 1024, device='cuda', dtype=torch.float32)
    return [p_matmul_weight, p_matmul_bias, p_bias, p_group_norm_weight, p_group_norm_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        p_matmul_weight, p_matmul_bias, p_bias, p_group_norm_weight, p_group_norm_bias, x = inputs
        
        # Reference computation
        import torch.nn.functional as F
        mm_ref = F.linear(x, p_matmul_weight, p_matmul_bias)
        swish_ref = torch.sigmoid(mm_ref) * mm_ref
        biased_ref = swish_ref + p_bias
        # GroupNorm: num_groups=64, num_channels=4096
        gn_ref = F.group_norm(biased_ref, 64, p_group_norm_weight, p_group_norm_bias, eps=1e-5)
        
        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - gn_ref).abs().max().item():.6e}")
        assert torch.allclose(output, gn_ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - gn_ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")