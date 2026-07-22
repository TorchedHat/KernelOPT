import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_8(arg_A, arg_B, out_ptr0):
    EVEN_K: tl.constexpr = False
    USE_FAST_ACCUM: tl.constexpr = False
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 16
    BLOCK_N: tl.constexpr = 32
    BLOCK_K: tl.constexpr = 16
    GROUP_M: tl.constexpr = 8
    ALLOW_TF32: tl.constexpr = False
    INDEX_DTYPE: tl.constexpr = tl.int32
    SPLIT_K: tl.constexpr = 8

    A = arg_A
    B = arg_B

    M = 2
    N = 96
    K = 24

    if M * N == 0:
        return

    stride_am = 24
    stride_ak = 1
    stride_bk = 1
    stride_bn = 24

    pid = tl.program_id(0).to(INDEX_DTYPE)
    pid_k = tl.program_id(2).to(INDEX_DTYPE)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Compute K slice for this CTA
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min(k_start + k_per_split, K)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_a_m = rm % M
    offs_b_n = rn % N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(k_per_split, BLOCK_K)):
        k_off = k_start + k_idx * BLOCK_K

        a_mask = (offs_k[None, :] + k_off) < k_end
        b_mask = (offs_k[:, None] + k_off) < k_end

        a_k_idx_vals = offs_k[None, :] + k_off
        b_k_idx_vals = offs_k[:, None] + k_off

        # A layout: stride_am=24, stride_ak=1 => index = k_offset + 24*m_offset
        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 24 * idx_m
        a = tl.load(A + xindex, mask=a_mask, other=0.0)

        # B layout: stride_bk=1, stride_bn=24 => index = k_offset + 24*n_offset
        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        b = tl.load(B + (idx_m + 24 * idx_n), mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # Write partial results via atomic_add (output was zeroed before launch)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    xindex = idx_n + N * idx_m
    tl.atomic_add(out_ptr0 + xindex, acc, mask=mask)

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
