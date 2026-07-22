import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=3),
], key=[], reset_to_zero=['out_ptr0'])
@triton.jit
def triton_tem_fused_convolution_79(in_ptr0, arg_A, arg_B, out_ptr0):
    # Split-K GEMM: SPLIT_K=16 gives grid=(16,16,1)=256 blocks on 132-SM GPU
    SPLIT_K: tl.constexpr = 16
    BLOCK_K: tl.constexpr = 64
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 32
    BLOCK_N: tl.constexpr = 32
    GROUP_M: tl.constexpr = 8
    ALLOW_TF32: tl.constexpr = False

    M = 490
    N = 32
    K = 832

    if M * N == 0:
        return

    # pid(0) = tile index in MxN grid (0..15)
    # pid(1) = split-K index (0..SPLIT_K-1)
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Compute the K range for this split
    K_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * K_per_split
    k_end = min(k_start + K_per_split, K)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_a_m = rm % M
    offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    num_k_iters = tl.cdiv(K_per_split, BLOCK_K)
    for k_idx in range(0, num_k_iters):
        k_off = k_start + k_idx * BLOCK_K
        mask_k = (offs_k + k_off) < k_end

        # A: shape [490, 832], stride_am=832, stride_ak=1
        # Load A[offs_a_m, offs_k + k_off]
        a = tl.load(arg_A + offs_a_m[:, None] * 832 + (offs_k[None, :] + k_off),
                    mask=mask_k[None, :], other=0.0)

        # B: original kernel loads B + (idx_m + 832*idx_n) where idx_m=k, idx_n=n
        # B has shape [32, 832] in memory (stride_bn=832, stride_bk=1)
        # B[k, n] = B_ptr + k + 832*n
        b = tl.load(arg_B + (offs_k[:, None] + k_off) + 832 * offs_b_n[None, :],
                    mask=mask_k[:, None], other=0.0)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # Atomic accumulate partial sums into zero-initialized output
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    xindex = idx_n + 32 * idx_m

    tl.atomic_add(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]),
                  acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (16, 1, 1)
    triton_tem_fused_convolution_79[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 832, 7, 7), (40768, 1, 5824, 832), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((490, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
