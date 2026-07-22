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
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 96
    K = 96
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = grid_m * grid_n
    num_programs = tl.num_programs(0)

    for tile_id in range(pid, total_tiles, num_programs):
        # re-order program ID for better L2 performance
        width = GROUP_M * grid_n
        group_id = tile_id // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
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
            # A: direct index (stride_am=96, stride_ak=1)
            a_offs = offs_a_m[:, None] * 96 + (offs_k[None, :] + k_idx * BLOCK_K)
            a = tl.load(A + a_offs)

            # B: direct index, NO double-broadcast
            # stride_bk=1, stride_bn=96 => B[k, n] = k*stride_bk + n*stride_bn = k + 96*n
            b_k = offs_k[:, None] + k_idx * BLOCK_K  # shape [BLOCK_K, 1]
            b_n = offs_b_n[None, :]                   # shape [1, BLOCK_N]
            b_offs = b_k + 96 * b_n                   # shape [BLOCK_K, BLOCK_N]
            b = tl.load(B + b_offs)

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

        # epilogue inside loop
        rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm2[:, None]
        idx_n = rn2[None, :]
        mask = (idx_m < M) & (idx_n < N)

        tmp0 = tl.load(in_ptr2 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr3 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr4 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp1 = acc - tmp0
        tmp3 = tl.full([1], 1e-05, tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tl.sqrt_rn(tmp4)
        tmp6 = tl.full([1], 1, tl.int32)
        tmp7 = (tmp6 / tmp5)
        tmp8 = tl.full([1], 1.0, tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp1 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tl.store(out_ptr1 + (idx_n + 96 * idx_m), tmp14, mask)

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
