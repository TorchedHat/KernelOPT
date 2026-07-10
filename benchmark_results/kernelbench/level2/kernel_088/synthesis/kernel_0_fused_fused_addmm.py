import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# GEMM kernel: computes out = x @ W.T + bias
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
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
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # W is [N, K], we want W[n, k] for out = x @ W.T
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


# GroupNorm reduction kernel: compute mean and variance per group
@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256, 'BLOCK_R': 32}, num_warps=4),
        triton.Config({'XBLOCK': 128, 'BLOCK_R': 32}, num_warps=4),
        triton.Config({'XBLOCK': 512, 'BLOCK_R': 32}, num_warps=8),
        triton.Config({'XBLOCK': 64, 'BLOCK_R': 32}, num_warps=2),
    ],
    key=['xnumel', 'rnumel'],
)
@triton.jit
def group_norm_reduction_kernel(
    In_ptr, Mean_ptr, Var_ptr,
    xnumel, rnumel,
    XBLOCK: tl.constexpr, BLOCK_R: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    
    r_index = tl.arange(0, BLOCK_R)[None, :]
    r_mask = r_index < rnumel
    
    # Load data: shape [XBLOCK, BLOCK_R]
    ptrs = In_ptr + xindex * rnumel + r_index
    data = tl.load(ptrs, mask=xmask & r_mask, other=0.0).to(tl.float32)
    
    # Compute mean
    count = tl.sum(r_mask.to(tl.float32), axis=1)[:, None]
    mean = tl.sum(data, axis=1)[:, None] / rnumel
    
    # Compute variance
    diff = data - mean
    var = tl.sum(diff * diff, axis=1)[:, None]
    
    tl.store(Mean_ptr + xindex, mean, mask=xmask)
    tl.store(Var_ptr + xindex, var, mask=xmask)


# Fused GroupNorm normalize + Swish + Multiply + Swish
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_norm_swish_kernel(
    In_ptr, Mean_ptr, Var_ptr,
    GN_Weight_ptr, GN_Bias_ptr, Mul_Weight_ptr,
    Out_ptr,
    total_elements, group_size, num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input
    x = tl.load(In_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute group index for each element
    # total_elements = batch * num_features
    # group_idx = (element_idx // group_size) for each element
    group_idx = offsets // group_size
    feat_idx = offsets % (num_groups * group_size)  # feature index within the row
    
    # Load mean and variance
    mean = tl.load(Mean_ptr + group_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(Var_ptr + group_idx, mask=mask, other=0.0).to(tl.float32)
    
    # Normalize
    inv_std = libdevice.rsqrt(var / group_size + eps)
    x_norm = (x - mean) * inv_std
    
    # Apply GroupNorm weight and bias
    feat_idx_in_row = offsets % (num_groups * group_size)
    gn_w = tl.load(GN_Weight_ptr + feat_idx_in_row, mask=mask, other=1.0).to(tl.float32)
    gn_b = tl.load(GN_Bias_ptr + feat_idx_in_row, mask=mask, other=0.0).to(tl.float32)
    x_gn = x_norm * gn_w + gn_b
    
    # Swish 1: x * sigmoid(x)
    sig1 = tl.sigmoid(x_gn)
    x_swish1 = x_gn * sig1
    
    # Multiply by weight
    mul_w = tl.load(Mul_Weight_ptr + feat_idx_in_row, mask=mask, other=1.0).to(tl.float32)
    x_mul = x_swish1 * mul_w
    
    # Swish 2: x * sigmoid(x)
    sig2 = tl.sigmoid(x_mul)
    x_out = x_mul * sig2
    
    tl.store(Out_ptr + offsets, x_out.to(tl.float32), mask=mask)


def kernel_function(p_gemm_weight, p_gemm_bias, p_group_norm_weight, p_group_norm_bias, p_multiply_weight, x):
    M, K = x.shape
    N = p_gemm_weight.shape[0]  # W is [out_features, in_features]
    
    # Allocate GEMM output
    gemm_out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    
    # Launch GEMM kernel
    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_kernel[grid_gemm](
        x, p_gemm_weight, p_gemm_bias, gemm_out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_gemm_weight.stride(0), p_gemm_weight.stride(1),
        gemm_out.stride(0), gemm_out.stride(1),
    )
    
    # GroupNorm parameters
    num_groups = 256
    num_features = N  # 8192
    group_size = num_features // num_groups  # 32
    
    # Total groups = batch * num_groups
    total_groups = M * num_groups  # 1024 * 256 = 262144
    
    # Allocate mean and var buffers
    mean_buf = torch.empty((total_groups,), device=x.device, dtype=torch.float32)
    var_buf = torch.empty((total_groups,), device=x.device, dtype=torch.float32)
    
    # Launch GroupNorm reduction
    xnumel = total_groups
    rnumel = group_size
    grid_gn = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)
    group_norm_reduction_kernel[grid_gn](
        gemm_out, mean_buf, var_buf,
        xnumel, rnumel,
    )
    
    # Allocate output
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    
    # Launch fused normalize + swish + multiply + swish
    total_elements = M * N
    eps = 1e-5
    grid_fused = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    fused_norm_swish_kernel[grid_fused](
        gemm_out, mean_buf, var_buf,
        p_group_norm_weight, p_group_norm_bias, p_multiply_weight,
        out,
        total_elements, group_size, num_groups,
        eps,
    )
    
    return out


def get_inputs():
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    num_groups = 256
    
    p_gemm_weight = torch.randn(out_features, in_features, device='cuda', dtype=torch.float32)
    p_gemm_bias = torch.randn(out_features, device='cuda', dtype=torch.float32)
    p_group_norm_weight = torch.randn(out_features, device='cuda', dtype=torch.float32)
    p_group_norm_bias = torch.randn(out_features, device='cuda', dtype=torch.float32)
    p_multiply_weight = torch.randn(out_features, device='cuda', dtype=torch.float32)
    x = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float32)
    
    return [p_gemm_weight, p_gemm_bias, p_group_norm_weight, p_group_norm_bias, p_multiply_weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        p_gemm_weight, p_gemm_bias, p_group_norm_weight, p_group_norm_bias, p_multiply_weight, x = inputs
        
        # Reference computation
        import torch.nn as nn
        linear = nn.Linear(8192, 8192, bias=True).cuda()
        linear.weight.data = p_gemm_weight
        linear.bias.data = p_gemm_bias
        
        group_norm = nn.GroupNorm(256, 8192).cuda()
        group_norm.weight.data = p_group_norm_weight
        group_norm.bias.data = p_group_norm_bias
        
        with torch.no_grad():
            ref = linear(x)
            ref = group_norm(ref)
            ref = ref * torch.sigmoid(ref)
            ref = ref * p_multiply_weight
            ref = ref * torch.sigmoid(ref)
        
        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")