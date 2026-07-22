import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=4,  num_stages=3),
        triton.Config({}, num_warps=4,  num_stages=4),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=3),
        triton.Config({}, num_warps=8,  num_stages=4),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=4),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused_convolution_relu_3(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 8388608
    N = 64
    K = 6
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 6
    stride_ak = 1
    stride_bk = 1
    stride_bn = 6

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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Preload full A tile: [BLOCK_M, 8] coalesced in K dim (stride_ak=1 is contiguous)
    # Use next power-of-2 >= K=6, which is 8
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 8)  # next power-of-2 >= K=6

    m_mask = rm < M
    n_mask = rn < N
    k_mask = rk < K  # K=6, so mask out indices 6 and 7

    # A tile: [BLOCK_M, 8], coalesced in K dim (stride_ak=1 is contiguous)
    a_tile = tl.load(
        A + rm[:, None] * stride_am + rk[None, :] * stride_ak,
        mask=m_mask[:, None] & k_mask[None, :],
        other=0.0
    )
    # B tile: [8, BLOCK_N], K-contiguous (stride_bk=1)
    b_tile = tl.load(
        B + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
        mask=k_mask[:, None] & n_mask[None, :],
        other=0.0
    )

    # Use tl.dot for the matrix multiply on the preloaded tiles
    # a_tile is [BLOCK_M, 8], b_tile is [8, BLOCK_N]
    # tl.dot requires both dims >= 16, so we need to pad to [BLOCK_M, 16] x [16, BLOCK_N]
    # Pad a_tile from [BLOCK_M, 8] to [BLOCK_M, 16] by concatenating zeros
    a_pad = tl.cat(a_tile, tl.zeros((BLOCK_M, 8), dtype=tl.float32), can_reorder=False)
    # Pad b_tile from [8, BLOCK_N] to [16, BLOCK_N] by concatenating zeros
    b_pad = tl.cat(b_tile, tl.zeros((8, BLOCK_N), dtype=tl.float32), can_reorder=False)

    acc += tl.dot(a_pad, b_pad, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # rematerialize rm and rn to save registers
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 64*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (131072, 1, 1)
    triton_tem_fused_convolution_relu_3[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 6, 256, 256), (393216, 1, 1536, 6), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8388608, 64), (64, 1), device='cuda:0', dtype=torch.float32)
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
