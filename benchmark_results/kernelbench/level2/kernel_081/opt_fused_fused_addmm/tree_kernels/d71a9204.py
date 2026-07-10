import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



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
def gemm_swish_epilogue_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
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

    # L2 cache-friendly swizzling
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A and B
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining

        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (mask_k[None, :]),
                    other=0.0,
                    eviction_policy='evict_first')
        b = tl.load(b_ptrs,
                    mask=(mask_k[:, None]) & (offs_n[None, :] < N),
                    other=0.0,
                    eviction_policy='evict_first')

        acc = tl.dot(a, b, acc, allow_tf32=False)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Load bias and add in FP32 (preserve precision for large bias values)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Swish: x * sigmoid(x) — keep in FP32 for precision near 0
    sigmoid_acc = tl.sigmoid(acc)
    acc = acc * sigmoid_acc

    # Divide by 2
    acc = acc * 0.5

    # Clamp to [-1, 1] in FP32 — necessary because swish(x)/2 can exceed [-1,1]
    # for large |x| (swish(x)/2 → x/2 as x → +inf)
    acc = tl.maximum(acc, -1.0)
    acc = tl.minimum(acc, 1.0)

    # Tanh directly in FP32 — eliminates the FP16 cast round-trip and the
    # redundant outer clamp (tanh output is always in (-1,1) by definition)
    acc = libdevice.tanh(acc)

    # Store result directly as FP32 — no outer clamp needed (tanh is bounded)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask, eviction_policy='evict_first')

def kernel_function(p_gemm_weight, p_gemm_bias, x):
    # x: [M, K], weight: [N, K] (linear weight is transposed), bias: [N]
    M, K = x.shape
    N = p_gemm_weight.shape[0]

    # Output tensor
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Weight is [N, K], we need A @ W^T = x @ weight^T
    # So B = weight^T: stride_bk = N (stride along K in weight), stride_bn = 1
    # weight shape: [N, K], strides: [K, 1]
    # We want to compute x @ weight.T
    # A = x: [M, K], stride_am=K, stride_ak=1
    # B = weight: [N, K], but we access it as [K, N] transposed
    # stride_bk = 1 (moving along K in weight's second dim)
    # stride_bn = K (moving along N in weight's first dim)

    stride_am = x.stride(0)
    stride_ak = x.stride(1)
    stride_bk = p_gemm_weight.stride(1)  # stride along K dimension
    stride_bn = p_gemm_weight.stride(0)  # stride along N dimension

    stride_cm = output.stride(0)
    stride_cn = output.stride(1)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    gemm_swish_epilogue_kernel[grid](
        x, p_gemm_weight, p_gemm_bias, output,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )

    return output


def get_inputs():
    p_gemm_weight = torch.randn(8192, 8192, dtype=torch.float32, device='cuda')
    p_gemm_bias = torch.randn(8192, dtype=torch.float32, device='cuda')
    x = torch.rand(1024, 8192, dtype=torch.float32, device='cuda')
    return [p_gemm_weight, p_gemm_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_gemm_weight, p_gemm_bias, x = inputs

        # Reference computation
        import torch.nn.functional as F
        ref = F.linear(x, p_gemm_weight, p_gemm_bias)
        ref = ref * torch.sigmoid(ref)
        ref = ref / 2.0
        ref = torch.clamp(ref, min=-1.0, max=1.0)
        ref = torch.tanh(ref)
        ref = torch.clamp(ref, min=-1.0, max=1.0)

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.6e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")