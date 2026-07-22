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
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_31(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 64
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32

    M = 2
    N = 384
    K = 1728
    grid_m = 1   # ceil(2/2)
    grid_n = 12  # ceil(384/32)
    num_tiles = 12  # grid_m * grid_n = 1 * 12

    num_programs = tl.num_programs(0)
    tile_id = tl.program_id(0).to(INDEX_DTYPE)

    while tile_id < num_tiles:
        pid_m = tile_id // grid_n   # always 0 since grid_m=1
        pid_n = tile_id % grid_n    # 0..11

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_a_m = rm % M
        offs_b_n = rn % N
        offs_k = tl.arange(0, BLOCK_K)

        # acc MUST be reset inside the while-loop for each tile
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            a_k = offs_k[None, :] + k_idx * BLOCK_K   # [1, BLOCK_K]
            b_k = offs_k[:, None] + k_idx * BLOCK_K   # [BLOCK_K, 1]

            # A: shape [M, K], stride [1728, 1] -> row-major
            xindex_a = a_k + 1728 * offs_a_m[:, None]  # [BLOCK_M, BLOCK_K]
            a = tl.load(arg_A + xindex_a)

            # B: shape [K, N], stride_bk=1, stride_bn=1728
            # xindex = idx_n + 384*idx_m where idx_m=b_k, idx_n=offs_b_n
            xindex_b = offs_b_n[None, :] + 384 * b_k   # [BLOCK_K, BLOCK_N]
            b = tl.load(arg_B + xindex_b)

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

        # Epilogue: batch norm (subtract mean, normalize, scale, shift)
        idx_m = rm[:, None]   # [BLOCK_M, 1]
        idx_n = rn[None, :]   # [1, BLOCK_N]
        mask = (idx_m < M) & (idx_n < N)

        tmp0  = tl.load(in_ptr2 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp2  = tl.load(in_ptr3 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr4 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp1  = acc - tmp0
        tmp3  = tl.full([1], 1e-05, tl.float32)
        tmp4  = tmp2 + tmp3
        tmp5  = tl.sqrt_rn(tmp4)
        tmp6  = tl.full([1], 1, tl.int32)
        tmp7  = (tmp6 / tmp5)
        tmp8  = tl.full([1], 1.0, tl.float32)
        tmp9  = tmp7 * tmp8
        tmp10 = tmp1 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 384 * idx_m, [BLOCK_M, BLOCK_N])), tmp14, mask)

        # Persistent stride: each SM takes next unprocessed tile
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    grid = (12, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_31[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((2, 1728, 1, 1), (1728, 1, 1728, 1728), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384, 1728, 1, 1), (1728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((2, 384, 1, 1), (384, 1, 384, 384), device='cuda:0', dtype=torch.float32)
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
