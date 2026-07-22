import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    NUM_TILES_PER_BATCH : tl.constexpr = 32
    TOTAL_WORK : tl.constexpr = 524288
    GRID_SIZE : tl.constexpr = 528

    M = 3
    N = 1024
    K = 3

    stride_aq = 9
    stride_am = 3
    stride_ak = 1

    stride_bq = 3072
    stride_bk = 1024
    stride_bn = 1

    worker_id = tl.program_id(0).to(INDEX_DTYPE)
    work_id = worker_id

    while work_id < TOTAL_WORK:
        idx_q = work_id // NUM_TILES_PER_BATCH
        pid = work_id % NUM_TILES_PER_BATCH

        # based on triton.ops.matmul
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N

        # re-order program ID for better L2 performance
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        ram = rm % M
        rbn = rn % N

        # Recompute A_ptr and B_ptr from base pointers each work item
        A_ptr = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq)
        B_ptr = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq)

        # CRITICAL: acc reset INSIDE the work-item loop — fixes prior correctness failures
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

        # rematerialize rm and rn to save registers
        rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm2[:, None]
        idx_n = rn2[None, :]
        mask = (idx_m < M) & (idx_n < N)

        # inductor generates a suffix
        xindex = idx_n + 1024 * idx_m + 3072 * idx_q
        tl.store(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask)

        work_id += GRID_SIZE

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (32, 1, 1)
    triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2048, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 3, 64, 16, 1), (24576, 3072, 1024, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16384, 3, 1024), (3072, 1024, 1), device='cuda:0', dtype=torch.float32)
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
