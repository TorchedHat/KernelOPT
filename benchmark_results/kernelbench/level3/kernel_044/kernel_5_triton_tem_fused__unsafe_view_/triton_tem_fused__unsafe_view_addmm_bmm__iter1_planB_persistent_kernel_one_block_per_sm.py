import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_5(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    NUM_SMS : tl.constexpr = 132

    M = 512
    N = 512
    K = 96

    stride_am = 96
    stride_ak = 1

    stride_bk = 512
    stride_bn = 1

    stride_aq_head  : tl.constexpr = 49152    # stride between heads within outer batch
    stride_aq_outer : tl.constexpr = 393216   # stride between outer batch items
    stride_bq_head  : tl.constexpr = 49152
    stride_bq_outer : tl.constexpr = 393216

    grid_m = 8   # M // BLOCK_M = 512 // 64
    grid_n = 8   # N // BLOCK_N = 512 // 64
    NUM_BATCH : tl.constexpr = 1024  # 128 outer * 8 heads
    tiles_per_batch = grid_m * grid_n  # 64
    total_tiles = tiles_per_batch * NUM_BATCH  # 65536

    sm_id = tl.program_id(0).to(INDEX_DTYPE)
    tile_id = sm_id

    while tile_id < total_tiles:
        # Decompose tile_id into batch index and local tile index
        idx_q = tile_id // tiles_per_batch
        local = tile_id % tiles_per_batch

        # L2-friendly GROUP_M swizzle within each batch slice
        width = GROUP_M * grid_n
        group_id = local // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (local % group_size)
        pid_n = (local % width) // group_size

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        outer_b = idx_q // 8   # outer batch index 0..127
        head_h  = idx_q % 8    # head index 0..7
        A_batch_offset = outer_b * stride_aq_outer + head_h * stride_aq_head
        B_batch_offset = outer_b * stride_bq_outer + head_h * stride_bq_head

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        # Use max_contiguous/multiple_of hints for contiguous access patterns
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

        A_ptr = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + A_batch_offset)
        B_ptr = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + B_batch_offset)

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

        # Store result
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)
        xindex = idx_n + 512 * idx_m + 262144 * idx_q
        tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

        tile_id += NUM_SMS

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (64, 1, 1)
    triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_5[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 8, 512, 96), (393216, 49152, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 8, 96, 512), (393216, 49152, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
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
