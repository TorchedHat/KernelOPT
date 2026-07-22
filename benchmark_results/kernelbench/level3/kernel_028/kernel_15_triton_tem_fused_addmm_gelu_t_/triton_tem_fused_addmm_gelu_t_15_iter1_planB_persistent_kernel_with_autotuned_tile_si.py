import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_gelu_t_15(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 256
    SPLIT_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 1
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 10
    K = 2048
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 2048
    stride_ak = 1
    stride_bk = 1
    stride_bn = 2048

    # based on triton.ops.matmul with split-K
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # Decompose pid into (pid_k, pid_mn)
    pid_k = pid % SPLIT_K
    pid_mn = pid // SPLIT_K

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid_mn // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_n = (pid_mn % width) // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_a_m = rm % M
    offs_b_n = rn % N

    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Compute K-slice for this split
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    for k_idx in range(0, tl.cdiv(k_per_split, BLOCK_K)):
        k_off = k_start + k_idx * BLOCK_K

        a_k_vals = offs_k[None, :] + k_off
        b_k_vals = offs_k[:, None] + k_off

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        idx_m_a = offs_a_m[:, None]
        idx_n_a = a_k_vals
        xindex_a = idx_n_a + 2048 * idx_m_a
        a_mask = (idx_m_a < M) & (idx_n_a < k_end)
        a = tl.load(A + xindex_a, mask=a_mask, other=0.0)

        # Load B tile: shape [BLOCK_K, BLOCK_N]
        # B has shape [N, K] with stride_bn=2048, stride_bk=1
        # B[k, n] => index = offs_b_n[None,:] + 2048*b_k_vals
        idx_m_b = b_k_vals
        idx_n_b = offs_b_n[None, :]
        xindex_b = idx_n_b + 2048 * idx_m_b
        b_mask = (idx_m_b < k_end) & (idx_n_b < N)
        b = tl.load(B + xindex_b, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # Use atomic_add to accumulate partial sums from different K-splits
    xindex = idx_n + 10 * idx_m
    tl.atomic_add(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (1, 1, 1)
    triton_tem_fused_addmm_gelu_t_15[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((10,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2, 10), (10, 1), device='cuda:0', dtype=torch.float32)
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
