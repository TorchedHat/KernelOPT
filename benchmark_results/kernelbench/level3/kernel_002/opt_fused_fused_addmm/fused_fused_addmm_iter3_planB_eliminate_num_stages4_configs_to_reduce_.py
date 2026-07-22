import torch
import triton
import triton.language as tl


# Fused GEMM + bias + ReLU kernel

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},  num_stages=3, num_warps=4),
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
    APPLY_BIAS: tl.constexpr,
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
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    if APPLY_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float32), mask=out_mask)

def run_gemm_bias_relu(A, B, bias, apply_relu=True):
    """
    Computes C = relu(A @ B.T + bias) or C = A @ B.T + bias
    """
    M, K = A.shape
    N = B.shape[0]
    assert B.shape[1] == K

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # B is (N, K), we want A @ B.T = (M, K) @ (K, N)
    # So we treat B as having stride_bk=1, stride_bn=K (column-major access for B.T)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    gemm_bias_relu_kernel[grid](
        A, B, bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(1), B.stride(0),  # transposed: stride_bk=B.stride(1), stride_bn=B.stride(0)
        C.stride(0), C.stride(1),
        APPLY_RELU=apply_relu,
        APPLY_BIAS=(bias is not None),
    )
    return C


def kernel_function(
    p_network_0_weight,
    p_network_0_bias,
    p_network_2_weight,
    p_network_2_bias,
    p_network_4_weight,
    p_network_4_bias,
    x,
):
    # Layer 0: linear(x, w0, b0) + relu
    # x: (128, 16384), w0: (32768, 16384) -> out: (128, 32768)
    h1 = run_gemm_bias_relu(x, p_network_0_weight, p_network_0_bias, apply_relu=True)

    # Layer 1: linear(h1, w2, b2) + relu
    # h1: (128, 32768), w2: (32768, 32768) -> out: (128, 32768)
    h2 = run_gemm_bias_relu(h1, p_network_2_weight, p_network_2_bias, apply_relu=True)

    # Layer 2: linear(h2, w4, b4) (no relu)
    # h2: (128, 32768), w4: (16384, 32768) -> out: (128, 16384)
    out = run_gemm_bias_relu(h2, p_network_4_weight, p_network_4_bias, apply_relu=False)

    return out


def get_inputs():
    p_network_0_weight = torch.randn(32768, 16384, dtype=torch.float32, device='cuda') * 0.01
    p_network_0_bias = torch.randn(32768, dtype=torch.float32, device='cuda') * 0.01
    p_network_2_weight = torch.randn(32768, 32768, dtype=torch.float32, device='cuda') * 0.01
    p_network_2_bias = torch.randn(32768, dtype=torch.float32, device='cuda') * 0.01
    p_network_4_weight = torch.randn(16384, 32768, dtype=torch.float32, device='cuda') * 0.01
    p_network_4_bias = torch.randn(16384, dtype=torch.float32, device='cuda') * 0.01
    x = torch.randn(128, 16384, dtype=torch.float32, device='cuda')
    return [
        p_network_0_weight,
        p_network_0_bias,
        p_network_2_weight,
        p_network_2_bias,
        p_network_4_weight,
        p_network_4_bias,
        x,
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Reference implementation
        p_network_0_weight, p_network_0_bias, p_network_2_weight, p_network_2_bias, \
            p_network_4_weight, p_network_4_bias, x = inputs

        h1_ref = torch.relu(x @ p_network_0_weight.T + p_network_0_bias)
        h2_ref = torch.relu(h1_ref @ p_network_2_weight.T + p_network_2_bias)
        out_ref = h2_ref @ p_network_4_weight.T + p_network_4_bias

        max_diff = (output - out_ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, out_ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")