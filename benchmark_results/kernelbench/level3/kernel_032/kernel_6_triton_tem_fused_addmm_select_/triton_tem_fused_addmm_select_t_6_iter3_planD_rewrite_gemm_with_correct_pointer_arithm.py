import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_select_t_6(in_ptr0, arg_A, arg_B, out_ptr0):
    BLOCK_M: tl.constexpr = 8
    BLOCK_N: tl.constexpr = 32
    BLOCK_K: tl.constexpr = 128
    GROUP_M: tl.constexpr = 8
    INDEX_DTYPE: tl.constexpr = tl.int32

    M = 10
    N = 1000
    K = 128

    if M * N == 0:
        return

    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Explicit pointer arithmetic — canonical Triton GEMM pattern
    # A[m, k] = arg_A + m*256 + k  (stride_am=256 because A is slice of (10,2,128))
    # B^T[k, n] = arg_B + k + n*128  (B is (1000,128) row-major, loaded as transposed)
    a_ptrs = arg_A + (offs_m[:, None] * 256 + offs_k[None, :])
    b_ptrs = arg_B + (offs_k[:, None] + offs_n[None, :] * 128)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N),
                    other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    idx_m = offs_m[:, None]
    idx_n = offs_n[None, :]
    mask = (idx_m < M) & (idx_n < N)

    bias = tl.load(in_ptr0 + idx_n,
                   mask=mask,
                   other=0.0,
                   eviction_policy='evict_last')
    tl.store(out_ptr0 + idx_m * N + idx_n, acc + bias, mask=mask)

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
