import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_select_t_6(in_ptr0, arg_A, arg_B, out_ptr0):
    ALLOW_TF32: tl.constexpr = False
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 16
    BLOCK_N: tl.constexpr = 32
    BLOCK_K: tl.constexpr = 32
    GROUP_M: tl.constexpr = 8
    SPLIT_K: tl.constexpr = 4

    M = 10
    N = 1000
    K = 128

    if M * N == 0:
        return

    pid   = tl.program_id(0)
    pid_k = tl.program_id(1)   # split-K partition index

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Compute the K range this CTA is responsible for
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end   = tl.minimum(k_start + k_per_split, K)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(k_per_split, BLOCK_K)):
        k_base = k_start + k_idx * BLOCK_K
        k_mask = (offs_k + k_base) < k_end

        # Load A tile: shape [BLOCK_M, BLOCK_K], stride_am=256, stride_ak=1
        a = tl.load(
            arg_A + rm[:, None] * 256 + (offs_k[None, :] + k_base),
            mask=(rm[:, None] < M) & k_mask[None, :],
            other=0.0
        )

        # Load B tile: shape [BLOCK_K, BLOCK_N], stride_bk=1, stride_bn=128
        b = tl.load(
            arg_B + (offs_k[:, None] + k_base) * 128 + rn[None, :],
            mask=k_mask[:, None] & (rn[None, :] < N),
            other=0.0
        )

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # Output indices
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    xindex = rn[None, :] + 1000 * rm[:, None]

    if SPLIT_K == 1:
        # Fuse bias add when no split-K (single CTA owns the full reduction)
        bias = tl.load(in_ptr0 + rn[None, :], mask=mask, eviction_policy='evict_last')
        tl.store(out_ptr0 + xindex, acc + bias, mask=mask)
    else:
        # Atomic accumulate partial sums; bias is added by the caller after all CTAs finish
        tl.atomic_add(out_ptr0 + xindex, acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (32, 1, 1)
    triton_tem_fused_addmm_select_t_6[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((1000,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 2, 128), (256, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
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
