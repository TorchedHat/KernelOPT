import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused_convolution_15(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 7840
    N = 32
    K = 192
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 192
    stride_ak = 1
    stride_bk = 1
    stride_bn = 192

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Create block pointers for A to handle strided 2D loads efficiently
    # A: shape=[M, K]=[7840, 192], strides=[stride_am, stride_ak]=[192, 1]
    # order=(1, 0) means innermost dim is K (stride_ak=1, contiguous)
    a_block_ptr = tl.make_block_ptr(
        base=A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # Explicit pointer arithmetic for B to avoid SMEM bank conflicts.
    # B has stride_bk=1 (K contiguous) and stride_bn=192 (N strided).
    # Using make_block_ptr with order=(1,0) caused 13.2-way bank conflicts.
    # Explicit 2D pointer array forces row-major [BLOCK_K, BLOCK_N] SMEM layout
    # with K as the fast dimension, matching global memory layout -> conflict-free.
    rk_base = tl.arange(0, BLOCK_K)  # [BLOCK_K], shape constexpr
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(INDEX_DTYPE)  # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
        # Build [BLOCK_K, BLOCK_N] pointer array for B
        # stride_bk=1 (K contiguous in global), stride_bn=192 (N strided)
        k_base = (k_idx * BLOCK_K).to(INDEX_DTYPE)
        b_ptrs = B + (k_base + rk_base[:, None]) * stride_bk + rn[None, :] * stride_bn
        b_mask = ((k_base + rk_base[:, None]) < K) & (rn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N], row-major SMEM = K-fast = conflict-free
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)  # a:[M,K] x b:[K,N], no tl.trans needed
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        # No b_block_ptr advance needed — k_base computed from k_idx each iteration

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 32*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (123, 1, 1)
    triton_tem_fused_convolution_15[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((7840, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    inputs = get_inputs()
    kernel_function(*inputs)
    torch.cuda.synchronize()
    if not args.profile_only:
        print("Kernel executed successfully.")
