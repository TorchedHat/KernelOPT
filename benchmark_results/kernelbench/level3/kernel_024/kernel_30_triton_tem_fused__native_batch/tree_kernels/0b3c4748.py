import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=5),
        triton.Config({}, num_warps=4, num_stages=5),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_30(arg_A, arg_B, out_ptr1):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 1728
    K = 432
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 432
    stride_ak = 1
    stride_bk = 1
    stride_bn = 432

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
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

    # Manual pointer arithmetic — restores 128 registers/thread vs 254 with tl.make_block_ptr
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k  = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B pointer TRANSPOSED: B[n, k] -- n-major, coalesced global load, conflict-free SMEM
    # offs_bn rows (N dim), offs_k cols (K dim)
    b_ptrs_T = B + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] + k_idx * BLOCK_K < K)
        b_mask_T = (offs_bn[:, None] < N) & (offs_k[None, :] + k_idx * BLOCK_K < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy='evict_first')
        b_T = tl.load(b_ptrs_T, mask=b_mask_T, other=0.0, eviction_policy='evict_first')
        # tl.trans(b_T) shape: (BLOCK_K, BLOCK_N) -- correct operand for dot
        acc += tl.dot(a, tl.trans(b_T), allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs_T += BLOCK_K * stride_bk  # advance K cols

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1728*idx_m
    tmp0 = tl.sigmoid(acc)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (54, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_30[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 432, 1, 1), (432, 1, 432, 432), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1728, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 1728, 1, 1), (1728, 1, 1728, 1728), device='cuda:0', dtype=torch.float32)
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
