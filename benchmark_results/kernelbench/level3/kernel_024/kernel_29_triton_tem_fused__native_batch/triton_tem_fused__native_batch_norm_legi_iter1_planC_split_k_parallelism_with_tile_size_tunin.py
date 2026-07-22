import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=[],
    reset_to_zero=['out_ptr1'],
)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_29(arg_A, arg_B, out_ptr1):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 1
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    SPLIT_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 432
    K = 1728
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1728
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1728

    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # Decode pid for split-K: pid_k selects K partition, pid_mn selects output tile
    pid_k = pid % SPLIT_K
    pid_mn = pid // SPLIT_K
    pid_m = pid_mn % grid_m
    pid_n = pid_mn // grid_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    # Compute K slice for this block
    K_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * K_per_split
    k_end = tl.minimum(k_start + K_per_split, K)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_a_m = rm % M
    offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    num_k_iters = tl.cdiv(K_per_split, BLOCK_K)
    for k_idx in range(0, num_k_iters):
        k_offset = k_start + k_idx * BLOCK_K

        a_k_vals = offs_k[None, :] + k_offset
        b_k_vals = offs_k[:, None] + k_offset

        # Mask for boundary on K
        k_mask_a = a_k_vals < k_end
        k_mask_b = b_k_vals < k_end

        idx_m = offs_a_m[:, None]
        idx_n = a_k_vals
        xindex = idx_n + 1728 * idx_m
        a = tl.load(A + xindex, mask=k_mask_a, other=0.0)

        idx_m = b_k_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 432 * idx_m
        b = tl.load(B + (tl.broadcast_to(idx_m + 1728 * idx_n, [BLOCK_K, BLOCK_N])),
                    mask=k_mask_b, other=0.0)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # Store partial results via atomic_add (no ReLU here — applied after all splits)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    xindex = idx_n + 432 * idx_m
    tl.atomic_add(out_ptr1 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (14, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_29[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 1728, 1, 1), (1728, 1, 1728, 1728), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((432, 1728, 1, 1), (1728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 432, 1, 1), (432, 1, 432, 432), device='cuda:0', dtype=torch.float32)
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
