import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 16, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128, 'SPLIT_K': 16, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 8, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 8, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 8, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 8, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 256, 'SPLIT_K': 16, 'GROUP_M': 8}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_split_k_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Number of tiles in M and N dimensions
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Grouped ordering for better L2 cache utilization
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # K split
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)
    
    # Offsets for A and B tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to A and B
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak)
    B_ptrs = B_ptr + ((k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over K dimension
    k_iters = tl.cdiv(k_end - k_start, BLOCK_K)
    
    for k_idx in range(0, k_iters):
        k_offset = k_start + k_idx * BLOCK_K
        
        # Boundary masks
        mask_m = offs_m[:, None] < M
        mask_k_a = (k_offset + offs_k[None, :]) < k_end
        mask_k_b = (k_offset + offs_k[:, None]) < k_end
        mask_n = offs_n[None, :] < N
        
        a = tl.load(A_ptrs, mask=mask_m & mask_k_a, other=0.0)
        b = tl.load(B_ptrs, mask=mask_k_b & mask_n, other=0.0)
        
        acc = tl.dot(a, b, acc=acc)
        
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    
    # Store result - use atomic add for split-K
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_cm = offs_cm[:, None] < M
    mask_cn = offs_cn[None, :] < N
    mask_c = mask_cm & mask_cn
    
    C_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    
    tl.atomic_add(C_ptrs, acc.to(tl.float32), mask=mask_c)


def kernel_function(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    
    # Output tensor - initialize to zero for atomic_add
    c = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    
    # Ensure contiguous
    a = a.contiguous()
    b = b.contiguous()
    
    def grid(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = triton.cdiv(N, META['BLOCK_N'])
        return (num_pid_m * num_pid_n, META['SPLIT_K'])
    
    matmul_split_k_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c


def get_inputs():
    M = 256
    N = 256
    K = 131072 * 4
    A = torch.rand(M, K, device='cuda', dtype=torch.float32)
    B = torch.rand(K, N, device='cuda', dtype=torch.float32)
    return [A, B]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    # Verify correctness
    if not args.profile_only:
        output = kernel_function(*inputs)
        ref = torch.matmul(inputs[0], inputs[1])
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff}")
        if max_diff < 1e-3:
            print("Correctness PASSED")
        else:
            print(f"Correctness FAILED: max_diff={max_diff}")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")