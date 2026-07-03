import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['BATCH', 'M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    BATCH, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # program ids
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache-friendly reordering
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # row/col offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # base pointers for this batch
    A = A_ptr + batch_id * stride_ab + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + batch_id * stride_bb + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # accumulate
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = rk < k_remaining

        a = tl.load(A, mask=(rm[:, None] < M) & (mask_k[None, :]), other=0.0)
        b = tl.load(B, mask=(mask_k[:, None]) & (rn[None, :] < N), other=0.0)

        acc = tl.dot(a, b, acc=acc, allow_tf32=False)

        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # store result
    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (cm[:, None] < M) & (cn[None, :] < N)

    C = C_ptr + batch_id * stride_cb + cm[:, None] * stride_cm + cn[None, :] * stride_cn
    tl.store(C, acc.to(tl.float32), mask=mask)


def kernel_function(a, b):
    # a: [BATCH, M, K], b: [BATCH, K, N]
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float32 and b.dtype == torch.float32

    BATCH, M, K = a.shape
    _, K2, N = b.shape
    assert K == K2

    c = torch.empty((BATCH, M, N), device=a.device, dtype=torch.float32)

    # strides
    stride_ab, stride_am, stride_ak = a.stride()
    stride_bb, stride_bk, stride_bn = b.stride()
    stride_cb, stride_cm, stride_cn = c.stride()

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
            BATCH,
        )

    bmm_kernel[grid](
        a, b, c,
        BATCH, M, N, K,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
    )

    return c


def get_inputs():
    batch_size = 128
    m = 512
    k = 1024
    n = 2048
    A = torch.rand(batch_size, m, k, device='cuda', dtype=torch.float32)
    B = torch.rand(batch_size, k, n, device='cuda', dtype=torch.float32)
    return [A, B]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify correctness
        ref = torch.bmm(inputs[0], inputs[1])
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference from reference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match PyTorch reference.")