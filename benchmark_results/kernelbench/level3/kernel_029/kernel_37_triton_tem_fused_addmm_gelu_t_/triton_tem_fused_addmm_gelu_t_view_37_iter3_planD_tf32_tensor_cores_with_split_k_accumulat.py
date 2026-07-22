import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=3, num_warps=4, maxnreg=128),
        triton.Config({}, num_stages=4, num_warps=4, maxnreg=128),
        triton.Config({}, num_stages=3, num_warps=8, maxnreg=128),
        triton.Config({}, num_stages=4, num_warps=8, maxnreg=128),
        triton.Config({}, num_stages=2, num_warps=4, maxnreg=128),
        triton.Config({}, num_stages=2, num_warps=8, maxnreg=128),
        triton.Config({}, num_stages=3, num_warps=4, maxnreg=96),
        triton.Config({}, num_stages=3, num_warps=8, maxnreg=96),
    ],
    key=[],
    reset_to_zero=['out_ptr0'],
)
@triton.jit
def triton_tem_fused_addmm_gelu_t_view_37(arg_A, arg_B, out_ptr0):
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 32
    M: tl.constexpr = 1960
    N: tl.constexpr = 384
    K: tl.constexpr = 1536
    GROUP_M: tl.constexpr = 8
    SPLIT_K: tl.constexpr = 8

    grid_m = tl.cdiv(M, BLOCK_M)   # 31
    grid_n = tl.cdiv(N, BLOCK_N)   # 3

    pid = tl.program_id(0)
    pid_k  = pid % SPLIT_K
    pid_mn = pid // SPLIT_K

    width = GROUP_M * grid_n
    group_id = pid_mn // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_n = (pid_mn % width) // group_size

    k_per_split = tl.cdiv(K, SPLIT_K)  # 192 when SPLIT_K=8
    k_start = pid_k * k_per_split

    # A is (M, K) row-major: strides (K=1536, 1)
    A_ptr = tl.make_block_ptr(
        base=arg_A, shape=(M, K), strides=(K, 1),
        offsets=(pid_m * BLOCK_M, k_start),
        block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    # B stored as (N=384, K=1536) row-major
    B_ptr = tl.make_block_ptr(
        base=arg_B, shape=(N, K), strides=(K, 1),
        offsets=(pid_n * BLOCK_N, k_start),
        block_shape=(BLOCK_N, BLOCK_K), order=(1, 0))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(tl.cdiv(k_per_split, BLOCK_K)):  # 6 iters when SPLIT_K=8
        a = tl.load(A_ptr, boundary_check=(0,))  # M=1960 not divisible by BLOCK_M=64
        b = tl.load(B_ptr)
        acc = tl.dot(a, tl.trans(b), acc, allow_tf32=True, out_dtype=tl.float32)
        A_ptr = tl.advance(A_ptr, (0, BLOCK_K))
        B_ptr = tl.advance(B_ptr, (0, BLOCK_K))

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.atomic_add(out_ptr0 + rm[:, None] * N + rn[None, :], acc, mask=mask)

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
