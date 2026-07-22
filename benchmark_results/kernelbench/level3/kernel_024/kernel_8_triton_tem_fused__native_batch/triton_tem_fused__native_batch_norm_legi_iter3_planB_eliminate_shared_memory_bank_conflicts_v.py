import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_8(arg_A, arg_B, out_ptr0):
    BLOCK_M: tl.constexpr = 2
    BLOCK_N: tl.constexpr = 16
    BLOCK_K: tl.constexpr = 16
    GROUP_M: tl.constexpr = 1
    ALLOW_TF32: tl.constexpr = False
    ACC_TYPE: tl.constexpr = tl.float32

    M = 2
    N = 96
    K = 24

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n

    cta_id = tl.program_id(0).to(tl.int32)
    num_ctas = tl.num_programs(0).to(tl.int32)

    tile_idx = cta_id
    while tile_idx < num_tiles:
        width = GROUP_M * grid_n
        group_id = tile_idx // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_idx % group_size)
        pid_n = (tile_idx % width) // group_size

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_a_m = rm % M
        offs_b_n = rn % N
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)

            a_k_vals = offs_k[None, :] + k_idx * BLOCK_K

            xindex_a = a_k_vals + 24 * offs_a_m[:, None]
            a = tl.load(arg_A + xindex_a, mask=a_mask, other=0.0)

            b_k_vals_T = offs_k[None, :] + k_idx * BLOCK_K
            b_mask_T = offs_k[None, :] < (K - k_idx * BLOCK_K)
            xindex_b_T = tl.broadcast_to(offs_b_n[:, None] * 24 + b_k_vals_T, [BLOCK_N, BLOCK_K])
            b_T = tl.load(arg_B + xindex_b_T, mask=b_mask_T, other=0.0)

            acc += tl.dot(a, tl.trans(b_T), allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

        rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m2 = rm2[:, None]
        idx_n2 = rn2[None, :]
        mask = (idx_m2 < M) & (idx_n2 < N)
        tl.store(out_ptr0 + idx_n2 + 96 * idx_m2, acc, mask=mask)

        tile_idx += num_ctas

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (3, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_8[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 24, 1, 1), (24, 1, 24, 24), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 96), (96, 1), device='cuda:0', dtype=torch.float32)
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
