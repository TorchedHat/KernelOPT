import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_convolution_1(arg_A, arg_B, out_ptr0):
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8

    M = 16777216
    N = 128
    K = 64

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                     # [BLOCK_K]

    # A[m,k]: shape [M,K], stride_am=64, stride_ak=1
    # A_ptrs[m,k] = arg_A + m*64 + k
    A_ptrs = arg_A + offs_m[:, None] * 64 + offs_k[None, :]  # [BLOCK_M, BLOCK_K]

    # B matrix: arg_B has shape [128, 64] with strides (64, 1)
    # In GEMM A[M,K] @ B_T[K,N]: B[k,n] = arg_B[n, k] = arg_B + n*64 + k*1
    # => B_ptrs[k,n] = arg_B + offs_k[:,None]*1 + offs_n[None,:]*64
    B_ptrs = arg_B + offs_k[:, None] * 1 + offs_n[None, :] * 64  # [BLOCK_K, BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptrs)  # EVEN_K=True (K=64 divisible by BLOCK_K=32), no mask needed
        b = tl.load(B_ptrs)  # EVEN_K=True, no mask needed
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
        A_ptrs += BLOCK_K      # advance K dim: stride_ak=1, so +BLOCK_K
        B_ptrs += BLOCK_K * 1  # advance K dim: stride_bk=1, so +BLOCK_K

    # rematerialize offs_m and offs_n to save registers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Output shape [16777216, 128], strides (128, 1)
    tl.store(out_ptr0 + offs_m[:, None] * 128 + offs_n[None, :], acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (131072, 1, 1)
    triton_tem_fused_convolution_1[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((16, 64, 1024, 1024), (67108864, 1, 65536, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16777216, 128), (128, 1), device='cuda:0', dtype=torch.float32)
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
