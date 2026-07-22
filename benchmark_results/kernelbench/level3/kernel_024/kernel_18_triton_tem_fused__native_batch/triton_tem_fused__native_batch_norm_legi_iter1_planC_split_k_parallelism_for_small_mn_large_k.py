import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_18(arg_A, arg_B, out_ptr1):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    SPLIT_K : tl.constexpr = 27

    A = arg_A
    B = arg_B

    M = 2
    N = 216
    K = 864

    if M * N == 0:
        return

    stride_am = 864
    stride_ak = 1
    stride_bk = 1
    stride_bn = 864

    # pid_mn: which (M, N) output tile; pid_sk: which K-split slice
    pid_mn = tl.program_id(0).to(INDEX_DTYPE)
    pid_sk = tl.program_id(1).to(INDEX_DTYPE)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

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

    # K slice for this split: K=864, SPLIT_K=27 => k_per_split=32 exactly
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_sk * k_per_split
    k_end = min(k_start + k_per_split, K)
    k_tiles = tl.cdiv(k_end - k_start, BLOCK_K)

    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, k_tiles):
        k_offset = k_start + k_idx * BLOCK_K

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_k_vals = offs_k[None, :] + k_offset
        idx_m_a = offs_a_m[:, None]
        idx_n_a = a_k_vals
        k_mask_a = (idx_n_a < k_end)
        xindex_a = idx_n_a + 864 * idx_m_a
        a = tl.load(A + xindex_a, mask=k_mask_a, other=0.0)

        # Load B tile: shape [BLOCK_K, BLOCK_N]
        b_k_vals = offs_k[:, None] + k_offset
        idx_m_b = b_k_vals
        idx_n_b = offs_b_n[None, :]
        k_mask_b = tl.broadcast_to(idx_m_b < k_end, [BLOCK_K, BLOCK_N])
        b_ptr = tl.broadcast_to(idx_m_b + 864 * idx_n_b, [BLOCK_K, BLOCK_N])
        b = tl.load(B + b_ptr, mask=k_mask_b, other=0.0)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # Atomically accumulate partial sums into out_ptr1 (used as workspace)
    rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m2 = rm2[:, None]
    idx_n2 = rn2[None, :]
    mask = (idx_m2 < M) & (idx_n2 < N)
    xindex = idx_n2 + N * idx_m2

    tl.atomic_add(out_ptr1 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (7, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_18[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 864, 1, 1), (864, 1, 864, 864), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((216, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 216, 1, 1), (216, 1, 216, 216), device='cuda:0', dtype=torch.float32)
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
