import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
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
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_14(arg_A, arg_B, out_ptr1):
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 2
    BLOCK_N: tl.constexpr = 32
    BLOCK_K: tl.constexpr = 32
    GROUP_M: tl.constexpr = 8

    A = arg_A
    B = arg_B

    M = 2
    N = 576
    K = 144

    if M * N == 0:
        return

    stride_am = 144
    stride_ak = 1
    stride_bk = 1
    stride_bn = 144

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_programs = tl.num_programs(0)

    tile_id = tl.program_id(0)

    while tile_id < num_tiles:
        # re-order program ID for better L2 performance
        width = GROUP_M * num_pid_n
        group_id = tile_id // width
        group_size = min(num_pid_m - group_id * GROUP_M, GROUP_M)
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
            a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
            b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)

            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

            idx_m = offs_a_m[:, None]
            idx_n = a_k_idx_vals
            xindex = idx_n + 144 * idx_m
            a = tl.load(A + xindex, mask=a_mask, other=0.0)

            idx_m = b_k_idx_vals
            idx_n = offs_b_n[None, :]
            xindex = idx_n + 576 * idx_m
            b = tl.load(B + (tl.broadcast_to(idx_m + 144 * idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape), mask=b_mask, other=0.0)

            # Broadcast MAC: avoids shared memory staging for BLOCK_M=2 < 16
            # a shape: (BLOCK_M, BLOCK_K), b shape: (BLOCK_K, BLOCK_N)
            # a[:, :, None] -> (BLOCK_M, BLOCK_K, 1), b[None, :, :] -> (1, BLOCK_K, BLOCK_N)
            # product -> (BLOCK_M, BLOCK_K, BLOCK_N), sum over axis=1 -> (BLOCK_M, BLOCK_N)
            acc += tl.sum(a[:, :, None].to(tl.float32) * b[None, :, :].to(tl.float32), axis=1)

        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)

        xindex = idx_n + 576 * idx_m
        tmp0 = tl.sigmoid(acc)
        tl.store(out_ptr1 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), tmp0, mask=mask)

        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (18, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_14[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 144, 1, 1), (144, 1, 144, 144), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((576, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 576, 1, 1), (576, 1, 576, 576), device='cuda:0', dtype=torch.float32)
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
