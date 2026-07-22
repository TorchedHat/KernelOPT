import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=2, num_stages=2),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22(arg_A, arg_B, out_ptr0):
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8

    M = 2250
    N = 480
    K = 80

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # A: shape [M, K], stride [K=80, 1] => A[m, k] = A_base + m*80 + k
    A_ptr = arg_A + rm[:, None] * 80 + rk[None, :]
    # B: stride_bk=1, stride_bn=80 => B[k, n] = B_base + k*1 + n*80
    B_ptr = arg_B + rk[:, None] * 1 + rn[None, :] * 80

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(A_ptr, mask=rk[None, :] < k_remaining, other=0.0)
        b = tl.load(B_ptr, mask=rk[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
        A_ptr += BLOCK_K
        B_ptr += BLOCK_K

    # rematerialize rm and rn to save registers
    rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm2[:, None] < M) & (rn2[None, :] < N)
    tl.store(out_ptr0 + rm2[:, None] * 480 + rn2[None, :], acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (144, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 80, 15, 15), (18000, 1, 1200, 80), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2250, 480), (480, 1), device='cuda:0', dtype=torch.float32)
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
