import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=1, num_stages=3),
        triton.Config({}, num_warps=1, num_stages=4),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_24(arg_A, arg_B, out_ptr1):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 1
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 288
    K = 1152
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    total_tiles = grid_m * grid_n  # = 9

    start_tile = tl.program_id(0).to(tl.int32)
    num_blocks = tl.num_programs(0).to(tl.int32)

    for tile_id in range(start_tile, total_tiles, num_blocks):
        # re-order program ID for better L2 performance
        width = GROUP_M * grid_n
        group_id = tile_id // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_a_m = rm % M
        offs_b_n = rn % N
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

            idx_m = offs_a_m[:, None]
            idx_n = a_k_idx_vals
            xindex = idx_n + 1152 * idx_m
            a = tl.load(A + xindex)

            idx_m = b_k_idx_vals
            idx_n = offs_b_n[None, :]
            xindex = idx_n + 288 * idx_m
            b = tl.load(B + (tl.broadcast_to(idx_m + 1152 * idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape))

            acc += tl.sum(a[:, :, None] * b[None, :, :], axis=1)

        # rematerialize rm and rn to save registers
        rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm2[:, None]
        idx_n = rn2[None, :]
        mask = (idx_m < M) & (idx_n < N)

        # inductor generates a suffix
        xindex = idx_n + 288 * idx_m
        tmp0 = tl.full([1], 0, tl.int32)
        tmp1 = tl.maximum(tmp0, acc)
        tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (9, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_24[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((288, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
