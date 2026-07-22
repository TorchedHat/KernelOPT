import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=8),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused_addmm_gelu_t_view_37(arg_A, arg_B, out_ptr0):
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 64
    M: tl.constexpr = 1960
    N: tl.constexpr = 384
    K: tl.constexpr = 1536
    GROUP_M: tl.constexpr = 8
    ALLOW_TF32: tl.constexpr = False
    ACC_TYPE: tl.constexpr = tl.float32

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Scalar pointer arithmetic — avoids tl.make_block_ptr boundary_check register overhead
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    offs_k = tl.arange(0, BLOCK_K)                    # (BLOCK_K,)

    # A_ptrs[m, k] = arg_A + m*K + k  — A is (M, K) row-major
    A_ptrs = arg_A + offs_m[:, None] * K + offs_k[None, :]  # (BLOCK_M, BLOCK_K)

    # B stored as (N=384, K=1536) row-major: B[n,k] = arg_B + n*K + k
    # B^T[k,n] = arg_B + offs_n[n]*K + offs_k[k]
    # Load (BLOCK_K, BLOCK_N) tile directly — no tl.trans needed!
    B_ptrs = arg_B + offs_n[None, :] * K + offs_k[:, None]  # (BLOCK_K, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(tl.cdiv(K, BLOCK_K)):
        # K=1536 is divisible by BLOCK_K=64, so k-dim mask is always True
        # Only M boundary mask needed (M=1960 not divisible by BLOCK_M=64)
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M), other=0.0)
        b = tl.load(B_ptrs, mask=(offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)  # no tl.trans!
        A_ptrs += BLOCK_K   # advance k: stride-1 in k for A
        B_ptrs += BLOCK_K   # advance k: stride-1 in k for B^T (B[n,k] stride in k is 1)
        offs_k += BLOCK_K

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptr0 + rm[:, None] * N + rn[None, :], acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (93, 1, 1)
    triton_tem_fused_addmm_gelu_t_view_37[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1960, 384), (384, 1), device='cuda:0', dtype=torch.float32)
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
