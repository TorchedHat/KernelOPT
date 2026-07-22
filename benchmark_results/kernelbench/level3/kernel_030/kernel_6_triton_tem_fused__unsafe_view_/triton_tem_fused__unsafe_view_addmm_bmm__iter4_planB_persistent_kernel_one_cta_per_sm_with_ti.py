import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__unsafe_view_addmm_bmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_6(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    NUM_SMS : tl.constexpr = 132
    NUM_HEADS : tl.constexpr = 3
    OUTER_BATCH : tl.constexpr = 640

    M = 49
    N = 49
    K = 32

    stride_aq = 1568
    stride_am = 32
    stride_ak = 1

    stride_bq = 1568
    stride_bk = 49
    stride_bn = 1

    # Persistent kernel: one CTA per SM
    sm_id = tl.program_id(0).to(INDEX_DTYPE)

    # Compute grid dimensions with BLOCK_M=32, BLOCK_N=32
    grid_m = (M + BLOCK_M - 1) // BLOCK_M  # = 2
    grid_n = (N + BLOCK_N - 1) // BLOCK_N  # = 2
    tiles_per_slice = grid_m * grid_n       # = 4
    total_tiles = tiles_per_slice * NUM_HEADS * OUTER_BATCH  # = 7680

    for work_id in range(sm_id, total_tiles, NUM_SMS):
        # Decode work_id into tile and batch indices
        tile_idx = work_id % tiles_per_slice
        batch_idx = work_id // tiles_per_slice
        pid_head = (batch_idx % NUM_HEADS).to(tl.int32)
        pid_outer = (batch_idx // NUM_HEADS).to(tl.int32)

        # Decode tile_idx to pid_m/pid_n with L2 grouping
        width = GROUP_M * grid_n
        group_id = tile_idx // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_idx % group_size)
        pid_n = (tile_idx % width) // group_size

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        # Use simple modulo for ram/rbn
        ram = rm % M
        rbn = rn % N

        # Compute pointer offsets
        A_ptr = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + pid_head * stride_aq + pid_outer * 4704)
        B_ptr = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + pid_head * stride_bq + pid_outer * 4704)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for k in range(K, 0, -BLOCK_K):
            if EVEN_K:
                a = tl.load(A_ptr)
                b = tl.load(B_ptr)
            else:
                a = tl.load(A_ptr, mask=rk[None, :] < k, other=0.)
                b = tl.load(B_ptr, mask=rk[:, None] < k, other=0.)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_ptr += BLOCK_K * stride_ak
            B_ptr += BLOCK_K * stride_bk

        # Store with correct output index
        idx_q = NUM_HEADS * pid_outer + pid_head  # = 3*pid_outer + pid_head
        rm_store = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn_store = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm_store[:, None]
        idx_n = rn_store[None, :]
        mask = (idx_m < M) & (idx_n < N)
        xindex = idx_n + 49 * idx_m + 2401 * idx_q
        tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (1, 1, 1)
    triton_tem_fused__unsafe_view_addmm_bmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_6[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((640, 3, 49, 32), (4704, 1568, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((640, 3, 32, 49), (4704, 1568, 49, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1920, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
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
