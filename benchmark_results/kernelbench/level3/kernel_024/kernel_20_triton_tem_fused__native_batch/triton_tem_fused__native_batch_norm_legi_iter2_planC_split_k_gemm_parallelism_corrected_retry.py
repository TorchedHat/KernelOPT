import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=1, num_stages=3),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_20(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    SPLIT_K : tl.constexpr = 22
    A = arg_A
    B = arg_B

    M = 2
    N = 192
    K = 864
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 864
    stride_ak = 1
    stride_bk = 1
    stride_bn = 864

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

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Compute this CTA's K slice bounds
    k_per_split = tl.cdiv(K, SPLIT_K)   # ceil(864/22) = 40
    k_start = pid_k * k_per_split
    k_end = min(k_start + k_per_split, K)  # clamp to K=864

    # Loop over only this CTA's K slice
    k_tiles = tl.cdiv(k_end - k_start, BLOCK_K)
    for k_idx in range(0, k_tiles):
        k_base = k_start + k_idx * BLOCK_K

        a_k_vals = offs_k[None, :] + k_base   # shape [1, BLOCK_K]
        b_k_vals = offs_k[:, None] + k_base   # shape [BLOCK_K, 1]

        # Tail mask: guard against k_base+offs_k >= k_end
        k_mask_a = a_k_vals < k_end   # shape [1, BLOCK_K]
        k_mask_b = b_k_vals < k_end   # shape [BLOCK_K, 1]

        idx_m = offs_a_m[:, None]     # shape [BLOCK_M, 1]
        idx_n = a_k_vals              # shape [1, BLOCK_K]
        xindex = idx_n + 864 * idx_m  # shape [BLOCK_M, BLOCK_K]
        a = tl.load(A + xindex, mask=k_mask_a, other=0.0)

        idx_m = b_k_vals              # shape [BLOCK_K, 1]
        idx_n = offs_b_n[None, :]     # shape [1, BLOCK_N]
        xindex_b = tl.broadcast_to(idx_m + 864 * idx_n, [BLOCK_K, BLOCK_N])
        b = tl.load(B + xindex_b, mask=tl.broadcast_to(k_mask_b, [BLOCK_K, BLOCK_N]), other=0.0)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 192 * idx_m
    # Use atomic_add so concurrent SPLIT_K CTAs accumulate correctly
    tl.atomic_add(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (6, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_20[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 864, 1, 1), (864, 1, 864, 864), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 192), (192, 1), device='cuda:0', dtype=torch.float32)
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
