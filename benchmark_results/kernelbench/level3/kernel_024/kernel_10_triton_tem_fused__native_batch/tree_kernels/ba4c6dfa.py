import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_10(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32

    M = 2
    N = 96
    K = 96
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    pid = tl.program_id(0).to(INDEX_DTYPE)
    num_programs = tl.num_programs(0).to(INDEX_DTYPE)

    grid_m = tl.cdiv(M, BLOCK_M)   # = 1
    grid_n = tl.cdiv(N, BLOCK_N)   # = 6
    total_tiles = grid_m * grid_n   # = 6

    tile_id = pid
    while tile_id < total_tiles:
        # L2-friendly GROUP_M reordering using tile_id (NOT pid)
        width = GROUP_M * grid_n
        group_id = tile_id // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_a_m = rm % M
        offs_b_n = rn % N
        offs_k = tl.arange(0, BLOCK_K)

        # acc MUST be initialized inside the loop (per tile)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            # A load: shape [BLOCK_M, BLOCK_K], stride_am=96, stride_ak=1
            a_ptrs = arg_A + offs_a_m[:, None] * 96 + (offs_k[None, :] + k_idx * BLOCK_K) * 1
            a = tl.load(a_ptrs)

            # B load: direct strided pointer, no double-broadcast, no bank conflicts
            # stride_bk=1 for K dimension, stride_bn=96 for N dimension
            b_ptrs = arg_B + (offs_k[:, None] + k_idx * BLOCK_K) * 1 + offs_b_n[None, :] * 96
            b = tl.load(b_ptrs)

            acc += tl.dot(a, b, allow_tf32=False, out_dtype=ACC_TYPE)

        # Epilogue: BN fused
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)

        tmp0 = tl.load(in_ptr2 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr3 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr4 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp1 = acc - tmp0
        tmp4 = tmp2 + tl.full([1], 1e-05, tl.float32)
        tmp5 = tl.sqrt_rn(tmp4)
        tmp9 = tl.full([1], 1.0, tl.float32) / tmp5
        tmp10 = tmp1 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tl.store(out_ptr1 + (idx_n + 96 * idx_m), tmp14, mask)

        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    grid = (3, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_10[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((2, 96, 1, 1), (96, 1, 96, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((2, 96, 1, 1), (96, 1, 96, 96), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6]

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
