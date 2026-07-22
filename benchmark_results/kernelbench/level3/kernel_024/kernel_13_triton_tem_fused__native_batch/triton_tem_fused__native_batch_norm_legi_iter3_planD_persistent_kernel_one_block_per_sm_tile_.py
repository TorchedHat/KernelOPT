import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=4),
    triton.Config({}, num_warps=8, num_stages=4),
], key=[])
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_13(arg_A, arg_B, out_ptr1):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 4
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 144
    K = 576
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    # Persistent tile loop
    pid = tl.program_id(0).to(INDEX_DTYPE)
    num_programs = tl.num_programs(0)
    grid_m = tl.cdiv(M, BLOCK_M)   # = 1
    grid_n = tl.cdiv(N, BLOCK_N)   # = 5
    total_tiles = grid_m * grid_n   # = 5

    tile_id = pid
    while tile_id < total_tiles:
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
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
            b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
            a_k_vals = offs_k[None, :] + k_idx * BLOCK_K
            b_k_vals = offs_k[:, None] + k_idx * BLOCK_K

            # A load: row-major [M, K], stride_am=576, stride_ak=1
            a = tl.load(A + offs_a_m[:, None] * 576 + a_k_vals, mask=a_mask, other=0.0)

            # B load FIXED: stride_bk=1, stride_bn=576 — N contiguous (no bank conflicts)
            # B[k, n] = B_ptr + n*576 + k
            b = tl.load(B + offs_b_n[None, :] * 576 + b_k_vals, mask=b_mask, other=0.0)

            acc += tl.dot(a, b, allow_tf32=False, out_dtype=ACC_TYPE)

        # Epilogue: ReLU + store
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)
        xindex = idx_n + 144 * idx_m
        tmp0 = tl.full([1], 0, tl.int32)
        tmp1 = tl.maximum(tmp0, acc)
        tl.store(out_ptr1 + xindex, tmp1, mask=mask)

        tile_id += num_programs  # advance to next tile

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (5, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_13[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 576, 1, 1), (576, 1, 576, 576), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((144, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 144, 1, 1), (144, 1, 144, 144), device='cuda:0', dtype=torch.float32)
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
