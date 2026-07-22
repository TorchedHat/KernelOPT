import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_29(arg_A, arg_B, out_ptr1):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 432
    K = 1728
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1728
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1728

    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_size = tl.num_programs(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M  # = 1
    grid_n = (N + BLOCK_N - 1) // BLOCK_N  # = 27
    num_tiles = grid_m * grid_n             # = 27

    # Persistent loop: each SM processes multiple tiles
    tile_id = pid
    while tile_id < num_tiles:
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
        # CRITICAL: keep % M and % N exactly as original for boundary-safe modular indexing
        offs_a_m = rm % M
        offs_b_n = rn % N
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

            idx_m = offs_a_m[:, None]
            idx_n = a_k_idx_vals
            xindex = idx_n + 1728*idx_m
            a = tl.load(A + (xindex))

            idx_m = b_k_idx_vals
            idx_n = offs_b_n[None, :]
            xindex = idx_n + 432*idx_m
            b = tl.load(B + ((tl.broadcast_to(idx_m + 1728*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)

        # inductor generates a suffix
        xindex = idx_n + 432*idx_m
        tmp0 = tl.full([1], 0, tl.int32)
        tmp1 = tl.maximum(tmp0, acc)
        tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

        tile_id += grid_size  # advance to next tile

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (14, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_29[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 1728, 1, 1), (1728, 1, 1728, 1728), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((432, 1728, 1, 1), (1728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 432, 1, 1), (432, 1, 432, 432), device='cuda:0', dtype=torch.float32)
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
