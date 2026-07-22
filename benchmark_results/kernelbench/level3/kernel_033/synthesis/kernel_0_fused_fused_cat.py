import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Kernel 1: Fused cat (concatenate x and hidden along dim=1)
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
def cat_kernel(x_ptr, hidden_ptr, out_ptr, batch_size, half_width, total_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    row = offsets // (2 * half_width)
    col = offsets % (2 * half_width)
    
    in_first_half = col < half_width
    
    x_idx = row * half_width + col
    h_idx = row * half_width + (col - half_width)
    
    x_val = tl.load(x_ptr + x_idx, mask=mask & in_first_half, other=0.0)
    h_val = tl.load(hidden_ptr + h_idx, mask=mask & ~in_first_half, other=0.0)
    
    result = tl.where(in_first_half, x_val, h_val)
    tl.store(out_ptr + offsets, result, mask=mask)


# Kernel 2: GEMM for i2h: [256, 32768] x [32768, 16384] -> [256, 16384]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_tanh_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k + k * BLOCK_K < K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    
    # Add bias and apply tanh
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    acc = libdevice.tanh(acc)
    
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Kernel 3: GEMM for h2o: [256, 16384] x [16384, 8192] -> [256, 8192] with bias
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k + k * BLOCK_K < K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def kernel_function(p_i2h_weight, p_i2h_bias, p_h2o_weight, p_h2o_bias, c_hidden, x, initial_hidden):
    # Step 1: Copy initial_hidden to c_hidden (both are [256, 16384])
    # c_hidden.copy_(initial_hidden) - we use initial_hidden as the hidden state
    hidden = initial_hidden.clone()
    
    batch_size = x.shape[0]
    input_size = x.shape[1]
    hidden_size = hidden.shape[1]
    
    # Step 2: Concatenate x and hidden along dim=1 -> [256, 32768]
    combined_size = input_size + hidden_size
    combined = torch.empty(batch_size, combined_size, device=x.device, dtype=x.dtype)
    
    total_elements = batch_size * combined_size
    grid_cat = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    cat_kernel[grid_cat](x, hidden, combined, batch_size, input_size, total_elements)
    
    # Step 3: Linear i2h: combined @ i2h_weight.T + i2h_bias -> [256, 16384] with tanh
    # i2h_weight is [16384, 32768], so i2h_weight.T is [32768, 16384]
    M = batch_size  # 256
    N = p_i2h_weight.shape[0]  # 16384
    K = p_i2h_weight.shape[1]  # 32768
    
    hidden_out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    grid_i2h = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_bias_tanh_kernel[grid_i2h](
        combined, p_i2h_weight, p_i2h_bias, hidden_out,
        M, N, K,
        combined.stride(0), combined.stride(1),
        p_i2h_weight.stride(1), p_i2h_weight.stride(0),  # B is weight transposed: stride_bk=stride along K in weight = stride(1), stride_bn=stride along N in weight = stride(0)
        hidden_out.stride(0), hidden_out.stride(1),
    )
    
    # Step 4: Linear h2o: hidden_out @ h2o_weight.T + h2o_bias -> [256, 8192]
    # h2o_weight is [8192, 16384], so h2o_weight.T is [16384, 8192]
    M2 = batch_size  # 256
    N2 = p_h2o_weight.shape[0]  # 8192
    K2 = p_h2o_weight.shape[1]  # 16384
    
    output = torch.empty(M2, N2, device=x.device, dtype=x.dtype)
    
    grid_h2o = lambda meta: (triton.cdiv(M2, meta['BLOCK_M']) * triton.cdiv(N2, meta['BLOCK_N']),)
    matmul_bias_kernel[grid_h2o](
        hidden_out, p_h2o_weight, p_h2o_bias, output,
        M2, N2, K2,
        hidden_out.stride(0), hidden_out.stride(1),
        p_h2o_weight.stride(1), p_h2o_weight.stride(0),
        output.stride(0), output.stride(1),
    )
    
    return output


def get_inputs():
    batch_size = 256
    input_size = 16384
    hidden_size = 16384
    output_size = 8192
    
    p_i2h_weight = torch.randn(hidden_size, input_size + hidden_size, device='cuda', dtype=torch.float32)
    p_i2h_bias = torch.randn(hidden_size, device='cuda', dtype=torch.float32)
    p_h2o_weight = torch.randn(output_size, hidden_size, device='cuda', dtype=torch.float32)
    p_h2o_bias = torch.randn(output_size, device='cuda', dtype=torch.float32)
    c_hidden = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float32)
    x = torch.randn(batch_size, input_size, device='cuda', dtype=torch.float32)
    initial_hidden = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float32)
    
    return [p_i2h_weight, p_i2h_bias, p_h2o_weight, p_h2o_bias, c_hidden, x, initial_hidden]


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
        p_i2h_weight, p_i2h_bias, p_h2o_weight, p_h2o_bias, c_hidden, x, initial_hidden = inputs
        
        # Reference computation
        hidden_ref = initial_hidden.clone()
        combined_ref = torch.cat((x, hidden_ref), dim=1)
        hidden_new_ref = torch.tanh(combined_ref @ p_i2h_weight.T + p_i2h_bias)
        output_ref = hidden_new_ref @ p_h2o_weight.T + p_h2o_bias
        
        assert torch.allclose(output, output_ref, rtol=1e-2, atol=1e-3), \
            f"Results don't match! max_diff={((output - output_ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")