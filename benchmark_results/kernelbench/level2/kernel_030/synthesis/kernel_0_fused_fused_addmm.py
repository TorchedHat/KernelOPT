import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# GEMM kernel: computes output = x @ W.T + bias
# x: [M, K], W: [N, K] (weight stored as [out, in]), output: [M, N]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
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
    
    # X: [M, K], access X[offs_m, offs_k]
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W: [N, K], access W[offs_n, offs_k] -> for W.T: W[offs_n, offs_k]
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
        
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    
    # Add bias
    offs_m_final = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_final = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    bias = tl.load(B_ptr + offs_n_final, mask=offs_n_final < N, other=0.0)
    acc = acc + bias[None, :]
    
    out_mask = (offs_m_final[:, None] < M) & (offs_n_final[None, :] < N)
    out_ptrs = Out_ptr + offs_m_final[:, None] * N + offs_n_final[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_stages=2, num_warps=16),
        triton.Config({'XBLOCK': 2}, num_stages=2, num_warps=16),
        triton.Config({'XBLOCK': 4}, num_stages=2, num_warps=16),
        triton.Config({'XBLOCK': 8}, num_stages=2, num_warps=16),
    ],
    key=['num_groups', 'group_size'],
)
@triton.jit
def group_norm_hardtanh_kernel(
    X_ptr, Mean_ptr, Var_ptr, GN_Weight_ptr, GN_Bias_ptr, Out_ptr,
    M, N, num_groups, group_size,
    eps,
    ht_min, ht_max,
    XBLOCK: tl.constexpr,
    GROUP_SIZE_BLOCK: tl.constexpr,
):
    # Each program handles XBLOCK rows, one group per row-group pair
    # Actually let's do: pid = (batch_idx * num_groups + group_idx)
    # Total programs = M * num_groups
    pid = tl.program_id(0)
    
    batch_idx = pid // num_groups
    group_idx = pid % num_groups
    
    if batch_idx >= M:
        return
    
    # Load elements for this (batch, group)
    base = batch_idx * N + group_idx * group_size
    offs = tl.arange(0, GROUP_SIZE_BLOCK)
    mask = offs < group_size
    
    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / group_size
    
    # Compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / group_size
    
    # Normalize
    inv_std = libdevice.rsqrt(var + eps)
    x_norm = diff * inv_std
    
    # Apply weight and bias
    weight_offs = group_idx * group_size + offs
    weight = tl.load(GN_Weight_ptr + weight_offs, mask=mask, other=1.0)
    bias = tl.load(GN_Bias_ptr + weight_offs, mask=mask, other=0.0)
    
    out = x_norm * weight + bias
    
    # HardTanh
    out = tl.maximum(out, ht_min)
    out = tl.minimum(out, ht_max)
    
    tl.store(Out_ptr + base + offs, out, mask=mask)


def kernel_function(p_gemm_weight, p_gemm_bias, p_group_norm_weight, p_group_norm_bias, x):
    M, K = x.shape
    N = p_gemm_weight.shape[0]  # out_features
    
    # GEMM output
    gemm_out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    
    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_kernel[grid_gemm](
        x, p_gemm_weight, p_gemm_bias, gemm_out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_gemm_weight.stride(0), p_gemm_weight.stride(1),
    )
    
    # GroupNorm + HardTanh
    num_groups = 16
    group_size = N // num_groups  # 512
    eps = 1e-5
    ht_min = -2.0
    ht_max = 2.0
    
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    
    # Find next power of 2 >= group_size
    GROUP_SIZE_BLOCK = triton.next_power_of_2(group_size)
    
    total_programs = M * num_groups
    grid_gn = (total_programs,)
    
    group_norm_hardtanh_kernel[grid_gn](
        gemm_out, None, None, p_group_norm_weight, p_group_norm_bias, out,
        M, N, num_groups, group_size,
        eps, ht_min, ht_max,
        GROUP_SIZE_BLOCK=GROUP_SIZE_BLOCK,
    )
    
    return out


def get_inputs():
    p_gemm_weight = torch.randn(8192, 8192, device='cuda', dtype=torch.float32)
    p_gemm_bias = torch.randn(8192, device='cuda', dtype=torch.float32)
    p_group_norm_weight = torch.randn(8192, device='cuda', dtype=torch.float32)
    p_group_norm_bias = torch.randn(8192, device='cuda', dtype=torch.float32)
    x = torch.randn(1024, 8192, device='cuda', dtype=torch.float32)
    return [p_gemm_weight, p_gemm_bias, p_group_norm_weight, p_group_norm_bias, x]