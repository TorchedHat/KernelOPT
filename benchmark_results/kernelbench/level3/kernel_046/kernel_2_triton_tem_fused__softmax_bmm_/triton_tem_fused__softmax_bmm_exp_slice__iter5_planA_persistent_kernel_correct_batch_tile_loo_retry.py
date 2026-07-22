import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__softmax_bmm_exp_slice_sub_transpose_view_2(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32

    M = 32
    N = 512
    K = 100

    stride_aq = 4800
    stride_am = 1
    stride_ak = 48

    stride_bq = 51200
    stride_bk = 512
    stride_bn = 1

    grid_m = (M + BLOCK_M - 1) // BLOCK_M  # = 1
    grid_n = (N + BLOCK_N - 1) // BLOCK_N  # = 4
    tiles_per_batch = grid_m * grid_n       # = 4
    total_tiles = 2048 * tiles_per_batch    # = 8192

    start_tile = tl.program_id(0).to(INDEX_DTYPE)
    grid_size = tl.num_programs(0).to(INDEX_DTYPE)

    tile_id = start_tile
    while tile_id < total_tiles:
        # Decode batch index and tile within batch
        idx_q = tile_id // tiles_per_batch
        tile_within_batch = tile_id % tiles_per_batch

        # GROUP_M swizzle (grid_m=1, so group_size=1, pid_m=0, pid_n=tile_within_batch)
        width = GROUP_M * grid_n  # 8*4=32
        group_id = tile_within_batch // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)  # min(1,8)=1
        pid_m = group_id * GROUP_M + (tile_within_batch % group_size)  # =0
        pid_n = (tile_within_batch % width) // group_size  # =tile_within_batch

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        # stride_am==1, stride_ak==48 (not M=32), so use else branch for ram
        # stride_bk==512 (==N), stride_bn==1, so use max_contiguous for rbn
        ram = rm % M
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

        # CRITICAL FIX 1: recompute from original base pointers each iteration (absolute offsets)
        A_ptr = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq)
        B_ptr = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq)

        # CRITICAL FIX 2: zero accumulator inside the loop for each tile
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k in range(K, 0, -BLOCK_K):
            a = tl.load(A_ptr, mask=rk[None, :] < k, other=0.)
            b = tl.load(B_ptr, mask=rk[:, None] < k, other=0.)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_ptr += BLOCK_K * stride_ak
            B_ptr += BLOCK_K * stride_bk

        # Store result
        idx_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
        idx_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        mask = (idx_m < M) & (idx_n < N)
        xindex = idx_n + 512 * idx_m + 16384 * idx_q
        tl.store(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask)

        tile_id += grid_size  # advance by grid stride

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (4, 1, 1)
    triton_tem_fused__softmax_bmm_exp_slice_sub_transpose_view_2[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((204800, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 100, 512), (51200, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 32, 512), (16384, 512, 1), device='cuda:0', dtype=torch.float32)
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
