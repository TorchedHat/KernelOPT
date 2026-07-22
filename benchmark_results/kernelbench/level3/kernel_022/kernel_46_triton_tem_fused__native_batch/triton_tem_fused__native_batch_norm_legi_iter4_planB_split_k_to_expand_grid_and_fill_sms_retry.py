import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_hardtanh_mean_relu_t_view_46(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    SPLIT_K : tl.constexpr = 8
    A = arg_A
    B = arg_B

    M = 10
    N = 1000
    K = 1280
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1280
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1280

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    pid_k = tl.program_id(1).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # K range for this split
    k_per_split = K // SPLIT_K  # = 160
    k_start = pid_k * k_per_split  # 0, 160, 320, ..., 1120

    # Row indices for A (M dimension)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Column indices for B (N dimension)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # K indices (within a BLOCK_K tile)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Iterate over the K split: k_per_split / BLOCK_K = 160 / 32 = 5 iterations
    for k_idx in tl.static_range(0, k_per_split // BLOCK_K):
        k_base = k_start + k_idx * BLOCK_K

        # Load A tile: shape (BLOCK_M, BLOCK_K)
        # A is (M=10, K=1280), stride_am=1280, stride_ak=1
        # offs_m rows, k_base+offs_k cols
        a_k = k_base + offs_k[None, :]  # (1, BLOCK_K)
        a_m = offs_m[:, None]           # (BLOCK_M, 1)
        # Mask for M boundary (K is always in-bounds within split)
        mask_a = (a_m < M)
        xindex_a = a_m * stride_am + a_k * stride_ak
        a = tl.load(A + xindex_a, mask=mask_a, other=0.0)

        # Load B tile: B is stored as (N=1000, K=1280), stride_bn=1280, stride_bk=1
        # We need B[k, n] = B_stored[n, k] (transposed view)
        # Load B_stored[offs_n, k_base+offs_k] -> shape (BLOCK_N, BLOCK_K), then transpose
        b_n = offs_n[:, None]           # (BLOCK_N, 1)
        b_k = k_base + offs_k[None, :]  # (1, BLOCK_K)
        # Mask for N boundary (K is always in-bounds within split)
        mask_b = (b_n < N)
        xindex_b = b_n * stride_bn + b_k * stride_bk
        b_T = tl.load(B + xindex_b, mask=mask_b, other=0.0)  # (BLOCK_N, BLOCK_K)
        b = tl.trans(b_T)  # (BLOCK_K, BLOCK_N)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # Epilogue: add bias only from pid_k==0, then atomic accumulate
    idx_m = offs_m[:, None]
    idx_n = offs_n[None, :]
    mask = (idx_m < M) & (idx_n < N)
    xindex = idx_n + 1000 * idx_m

    if pid_k == 0:
        tmp0 = tl.load(in_ptr0 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        acc = acc + tmp0

    tl.atomic_add(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (32, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_hardtanh_mean_relu_t_view_46[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((1000,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 1280, 1, 1), (1280, 1, 12800, 12800), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
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
