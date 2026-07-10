import torch
import triton
import triton.language as tl

# GEMM kernel: computes out = x @ weight.T, shape [M, N]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    GROUP_M: tl.constexpr = 8
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + offs_k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_N': 4096}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_N': 8192}, num_stages=2, num_warps=32),
    ],
    key=['M', 'N'],
)
@triton.jit

def fused_div_sum_mul_kernel(
    inp_ptr, out_ptr,
    M, N,
    scaling_factor,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(inp_ptr + row * N + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        acc += tl.sum(x, axis=0)

    # divide by 2 and multiply by scaling_factor: combined = * (scaling_factor / 2)
    result = acc * (scaling_factor * 0.5)
    tl.store(out_ptr + row, result)


def kernel_function(p_weight, x):
    M, K = x.shape
    N = p_weight.shape[0]  # weight is [hidden_size, input_size], weight.T is [input_size, hidden_size]

    # Allocate intermediate buffer for GEMM output
    gemm_out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Launch GEMM: x @ weight.T
    # x: [M, K], weight: [N, K] (so weight.T is [K, N])
    # A = x (stride_am=K, stride_ak=1)
    # B = weight (stride_bk=1, stride_bn=K) -- accessing as transposed
    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_kernel[grid_gemm](
        x, p_weight, gemm_out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_weight.stride(1), p_weight.stride(0),  # transposed access
        gemm_out.stride(0), gemm_out.stride(1),
    )

    # Allocate output for reduction
    out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

    scaling_factor = 1.5

    grid_red = (M,)
    fused_div_sum_mul_kernel[grid_red](
        gemm_out, out,
        M, N,
        scaling_factor,
    )

    return out


def get_inputs():
    p_weight = torch.randn(8192, 8192, device='cuda', dtype=torch.float32)
    x = torch.rand(1024, 8192, device='cuda', dtype=torch.float32)
    return [p_weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_weight, x = inputs
        # Reference computation
        ref_mm = torch.matmul(x, p_weight.T)
        ref_div = ref_mm / 2
        ref_sum = torch.sum(ref_div, dim=1, keepdim=True)
        ref_out = ref_sum * 1.5

        assert torch.allclose(output, ref_out, rtol=1e-3, atol=1e-2), \
            f"Results don't match! max_diff={((output - ref_out).abs().max().item()):.4e}"
        print("Kernel executed successfully.")