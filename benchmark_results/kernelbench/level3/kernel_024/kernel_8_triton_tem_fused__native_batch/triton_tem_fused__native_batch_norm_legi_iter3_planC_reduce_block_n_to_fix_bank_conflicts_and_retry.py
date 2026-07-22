import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_8(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 1
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    NUM_SMS : tl.constexpr = 132
    A = arg_A
    B = arg_B

    M = 2
    N = 96
    K = 24
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 24
    stride_ak = 1
    stride_bk = 1
    stride_bn = 24

    grid_m = tl.cdiv(M, BLOCK_M)  # = 1
    grid_n = tl.cdiv(N, BLOCK_N)  # = 6
    num_tiles = grid_m * grid_n   # = 6

    sm_id = tl.program_id(0).to(INDEX_DTYPE)
    tile_id = sm_id

    while tile_id < num_tiles:
        pid_m = tile_id // grid_n
        pid_n = tile_id % grid_n

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_a_m = rm % M
        offs_b_n = rn % N
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
            b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)

            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

            idx_m = offs_a_m[:, None]
            idx_n = a_k_idx_vals
            a = tl.load(A + (idx_n + 24 * idx_m), mask=a_mask, other=0.0)

            idx_m = b_k_idx_vals
            idx_n = offs_b_n[None, :]
            b = tl.load(B + tl.broadcast_to(idx_m + 24 * idx_n, [BLOCK_K, BLOCK_N]), mask=b_mask, other=0.0)

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

        # rematerialize rm and rn to save registers
        rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm2[:, None]
        idx_n = rn2[None, :]
        mask = (idx_m < M) & (idx_n < N)

        xindex = idx_n + 96 * idx_m
        tl.store(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask=mask)

        tile_id = tile_id + NUM_SMS

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
