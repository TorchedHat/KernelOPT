import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
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
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_stages=2, num_warps=16),
    ],
    key=['M', 'N'],
)
@triton.jit
def epilogue_kernel(
    M, N,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row
    row = tl.program_id(0)
    if row >= M:
        return

    # Compute mean of (gemm_out + bias - subtract) over N
    mean_acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        gemm_val = tl.load(gemm_out_ptr + row * N + cols, mask=mask, other=0.0)
        bias_val = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        sub_val = tl.load(subtract_ptr + cols, mask=mask, other=0.0)
        val = gemm_val + bias_val - sub_val
        mean_acc += tl.sum(tl.where(mask, val, 0.0), axis=0)
    mean_val = mean_acc / N

    # logsumexp of [mean_val] along dim=1 with keepdim=True
    # Since input is shape [2048, 1], logsumexp over dim=1 of a single element = that element
    lse_val = mean_val

    # GELU
    gelu_val = lse_val * 0.5 * (1.0 + libdevice.erf(lse_val * 0.7071067811865476))

    # Residual add: gelu_val + original_x (broadcast gelu_val across N)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        orig = tl.load(original_x_ptr + row * N + cols, mask=mask, other=0.0)
        result = gelu_val + orig
        tl.store(out_ptr + row * N + cols, result, mask=mask)


def kernel_function(p_gemm_weight, p_gemm_bias, p_subtract, x):
    M, K = x.shape
    N, K2 = p_gemm_weight.shape
    assert K == K2

    # Step 1: GEMM (x @ W.T) using tl.dot
    gemm_out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_kernel[grid_gemm](
        x, p_gemm_weight, gemm_out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_gemm_weight.stride(1), p_gemm_weight.stride(0),  # B is W, we want x @ W.T so stride_bk=W.stride(1), stride_bn=W.stride(0)
        gemm_out.stride(0), gemm_out.stride(1),
    )

    # Step 2: Fused epilogue
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid_epilogue = (M,)
    epilogue_kernel[grid_epilogue](
        gemm_out, p_gemm_bias, p_subtract, x, out,
        M, N,
    )

    return out


def get_inputs():
    p_gemm_weight = torch.randn(8192, 8192, device='cuda', dtype=torch.float32)
    p_gemm_bias = torch.randn(8192, device='cuda', dtype=torch.float32)
    p_subtract = torch.randn(8192, device='cuda', dtype=torch.float32)
    x = torch.rand(2048, 8192, device='cuda', dtype=torch.float32)
    return [p_gemm_weight, p_gemm_bias, p_subtract, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    if not args.profile_only:
        p_gemm_weight, p_gemm_bias, p_subtract, x = inputs
        # Reference computation
        import torch.nn.functional as F
        original_x = x.clone().detach()
        ref_linear = F.linear(x, p_gemm_weight, p_gemm_bias)
        ref_sub = ref_linear - p_subtract
        ref_mean = torch.mean(ref_sub, dim=1, keepdim=True)
        ref_lse = torch.logsumexp(ref_mean, dim=1, keepdim=True)
        ref_gelu = F.gelu(ref_lse)
        ref_out = ref_gelu + original_x
        assert torch.allclose(output, ref_out, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref_out).abs().max().item()):.4e}"
        print("Kernel executed successfully.")