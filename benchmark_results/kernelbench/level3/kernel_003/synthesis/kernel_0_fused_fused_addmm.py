import torch
import triton
import triton.language as tl

# GEMM kernel with fused bias + ReLU epilogue
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
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
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    # Apply ReLU if needed
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)
    
    # Store
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float32), mask=c_mask)


def run_gemm_bias_relu(A, B, bias, apply_relu=True):
    """Run GEMM: C = A @ B.T + bias, optionally with ReLU"""
    M, K = A.shape
    N = B.shape[0]  # B is (N, K), we compute A @ B.T
    
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    gemm_bias_relu_kernel[grid](
        A, B, bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(1), B.stride(0),  # B transposed: stride_bk=B.stride(1), stride_bn=B.stride(0)
        C.stride(0), C.stride(1),
        APPLY_RELU=apply_relu,
    )
    return C


def kernel_function(
    p_network_0_weight, p_network_0_bias,
    p_network_2_weight, p_network_2_bias,
    p_network_4_weight, p_network_4_bias,
    p_network_6_weight, p_network_6_bias,
    p_network_8_weight, p_network_8_bias,
    p_network_10_weight, p_network_10_bias,
    p_network_12_weight, p_network_12_bias,
    p_network_14_weight, p_network_14_bias,
    p_network_16_weight, p_network_16_bias,
    p_network_18_weight, p_network_18_bias,
    p_network_20_weight, p_network_20_bias,
    p_network_22_weight, p_network_22_bias,
    p_network_24_weight, p_network_24_bias,
    p_network_26_weight, p_network_26_bias,
    p_network_28_weight, p_network_28_bias,
    p_network_30_weight, p_network_30_bias,
    p_network_32_weight, p_network_32_bias,
):
    # Layer 0: x (1024, 8192) @ w0.T (8192, 1024) + b0 -> (1024, 1024), ReLU
    h = run_gemm_bias_relu(x, p_network_0_weight, p_network_0_bias, apply_relu=True)
    # Layers 2-30: (1024, 1024) @ w.T (1024, 1024) + b -> (1024, 1024), ReLU
    h = run_gemm_bias_relu(h, p_network_2_weight, p_network_2_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_4_weight, p_network_4_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_6_weight, p_network_6_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_8_weight, p_network_8_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_10_weight, p_network_10_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_12_weight, p_network_12_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_14_weight, p_network_14_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_16_weight, p_network_16_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_18_weight, p_network_18_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_20_weight, p_network_20_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_22_weight, p_network_22_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_24_weight, p_network_24_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_26_weight, p_network_26_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_28_weight, p_network_28_bias, apply_relu=True)
    h = run_gemm_bias_relu(h, p_network_30_weight, p_network_30_bias, apply_relu=True)
    # Layer 32: (1024, 1024) @ w32.T (1024, 8192) + b32 -> (1024, 8192), no ReLU
    h = run_gemm_bias_relu(h, p_network_32_weight, p_network_32_bias, apply_relu=False)
    return h


def get_inputs():
    batch_size = 1024
    input_size = 8192
    hidden_size = 1024
    output_size = 8192
    
    p_network_0_weight = torch.randn(hidden_size, input_size, dtype=torch.float32, device='cuda') * 0.01
    p_network_0_bias = torch.randn(hidden_size, dtype=torch.float32, device='cuda') * 0.01
    
    weights_biases = []
    for i in range(15):
        weights_biases.append(torch.randn(hidden_size, hidden_size, dtype=torch.float32, device='cuda') * 0.01)
        weights_biases.append(torch.randn(hidden_size, dtype=torch.float32, device='cuda') * 0.01)
    
    p_network_32_weight = torch.randn(output_size, hidden_size, dtype=torch.float32, device='cuda') * 0.01
    p_network_32_bias = torch.randn(output_size, dtype=torch.float32, device='cuda') * 0.01
    
    x = torch.randn(batch_size, input_size, dtype=torch.float32, device='cuda') * 0.01
    
    return [p_network_0_weight, p_network_0_bias] + weights_biases + [p_network_32_weight, p_network_32_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        import torch.nn as nn
        
        # Build reference model
        batch_size = 1024
        input_size = 8192
        hidden_layer_sizes = [1024] * 16
        output_size = 8192
        
        # Extract weights from inputs
        (p_network_0_weight, p_network_0_bias,
         p_network_2_weight, p_network_2_bias,
         p_network_4_weight, p_network_4_bias,
         p_network_6_weight, p_network_6_bias,
         p_network_8_weight, p_network_8_bias,
         p_network_10_weight, p_network_10_bias,
         p_network_12_weight, p_network_12_bias,
         p_network_14_weight, p_network_14_bias,
         p_network_16_weight, p_network_16_bias,
         p_network_18_weight, p_network_18_bias,
         p_network_20_weight, p_network_20_bias,
         p_network_22_weight, p_network_22_bias,
         p_network_24_weight, p_network_24_bias,
         p_network_26_weight, p_network_26_bias,
         p_network_28_weight, p_network_28_bias,
         p_network_30_weight, p_network_30_bias,
         p_network_32_weight, p_network_32_bias,
         x) = inputs
        
        # Reference computation
        import torch.nn.functional as F
        h = x
        weights = [p_network_0_weight, p_network_2_weight, p_network_4_weight,
                   p_network_6_weight, p_network_8_weight, p_network_10_weight,
                   p_network_12_weight, p_network_14_weight, p_network_16_weight,
                   p_network_18_weight, p_network_20_weight, p_network_22_weight,
                   p_network_24_weight, p_network_26_weight, p_network_28_weight,
        biases = [p_network_0_bias, p_network_2_bias, p_network_4_bias,
                  p_network_6_bias, p_network_8_bias, p_network_10_bias,
                  p_network_12_bias, p_network_14_bias, p_network_16_bias,
                  p_network_18_bias, p_network_20_bias, p_network_22_bias,
                  p_network_24_bias, p_network_26_bias, p_network_28_bias,
        
        for i, (w, b) in enumerate(zip(weights, biases)):
            h = F.linear(h, w, b)
            if i < len(weights) - 1:
                h = F.relu(h)
        ref = h
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")