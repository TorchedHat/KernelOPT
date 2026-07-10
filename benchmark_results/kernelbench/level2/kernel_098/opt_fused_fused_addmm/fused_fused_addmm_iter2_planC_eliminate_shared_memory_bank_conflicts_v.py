import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# GEMM kernel: computes x @ W^T + bias, output shape [M, N]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
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

    # Store result
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 512}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 128}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512}, num_stages=3, num_warps=8),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 256}, num_stages=3, num_warps=8),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit

def pool_gelu_scale_max_kernel(
    in_ptr,
    out_max_ptr,
    xnumel,
    r0_numel,
    POOL_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    IN_COLS: tl.constexpr,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel

    r0_base = tl.arange(0, R0_BLOCK)[None, :]

    _max_acc = tl.full([XBLOCK, R0_BLOCK], float('-inf'), dtype=tl.float32)

    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel

        # avg_pool: sum 16 elements starting at r0_index * POOL_SIZE
        base_col = r0_index * POOL_SIZE  # [1, R0_BLOCK]

        # Load 16 elements and sum them
        s = tl.zeros([XBLOCK, R0_BLOCK], dtype=tl.float32)
        for i in range(POOL_SIZE):
            col = base_col + i
            ptr = in_ptr + xindex * IN_COLS + col
            val = tl.load(ptr, mask=xmask & r0_mask & (col < IN_COLS), other=0.0)
            s = s + val

        avg = s * (1.0 / POOL_SIZE)

        # GELU: x * 0.5 * (1 + erf(x * 0.7071067811865476))
        half_avg = avg * 0.5
        erf_val = libdevice.erf(avg * 0.7071067811865476)
        gelu_val = half_avg * (1.0 + erf_val)

        # Scale
        scaled = gelu_val * SCALE

        # Max reduction
        cur = tl.broadcast_to(scaled, [XBLOCK, R0_BLOCK])
        _max_acc = tl.where(xmask & r0_mask, tl.maximum(_max_acc, cur), _max_acc)

    # Reduce along r0 dimension
    row_max = tl.max(_max_acc, axis=1)[:, None]
    tl.store(out_max_ptr + xindex, row_max, mask=xmask)


def kernel_function(p_matmul_weight, p_matmul_bias, x):
    M, K = x.shape
    N = p_matmul_weight.shape[0]  # weight is [out_features, in_features]

    # Allocate linear output
    linear_out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Launch GEMM: x [M, K] @ W^T [K, N] + bias [N]
    # W is [N, K], so W^T is [K, N]
    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_bias_kernel[grid_gemm](
        x, p_matmul_weight, p_matmul_bias, linear_out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_matmul_weight.stride(1), p_matmul_weight.stride(0),  # B is W^T: stride_bk=W.stride(1), stride_bn=W.stride(0)
        linear_out.stride(0), linear_out.stride(1),
    )

    # Pool size and output size
    POOL_SIZE = 16
    pooled_N = N // POOL_SIZE  # 8192 // 16 = 512
    SCALE = 2.0

    # Allocate max output
    max_out = torch.empty((M,), device=x.device, dtype=x.dtype)

    # Launch fused pool+gelu+scale+max kernel
    grid_pool = lambda meta: (triton.cdiv(M, meta['XBLOCK']),)
    pool_gelu_scale_max_kernel[grid_pool](
        linear_out,
        max_out,
        M,
        pooled_N,
        POOL_SIZE,
        SCALE,
        N,
    )

    return max_out


def get_inputs():
    p_matmul_weight = torch.randn(8192, 8192, dtype=torch.float32, device='cuda')
    p_matmul_bias = torch.randn(8192, dtype=torch.float32, device='cuda')
    x = torch.randn(1024, 8192, dtype=torch.float32, device='cuda')
    return [p_matmul_weight, p_matmul_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_matmul_weight, p_matmul_bias, x = inputs

        # Reference computation
        import torch.nn.functional as F
        linear_ref = F.linear(x, p_matmul_weight, p_matmul_bias)
        pooled_ref = F.avg_pool1d(linear_ref.unsqueeze(1), kernel_size=16).squeeze(1)
        gelu_ref = F.gelu(pooled_ref)
        scaled_ref = gelu_ref * 2.0
        max_ref = torch.max(scaled_ref, dim=1).values

        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - max_ref).abs().max().item():.4e}")
        assert torch.allclose(output, max_ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={(output - max_ref).abs().max().item():.4e}"
        print("Kernel executed successfully.")