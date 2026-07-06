import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused_convolution_1(arg_A, arg_B, out_ptr0):
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8

    M = 16777216
    N = 128
    K = 64

    pid = tl.program_id(0).to(tl.int64)
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
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_idx * BLOCK_K

        # A load: stride_am=64, stride_ak=1  -> A[m, k] = arg_A + m*64 + k
        a_m = rm[:, None]
        a_k = (offs_k + k_start)[None, :]
        a = tl.load(arg_A + a_m * 64 + a_k * 1,
                    mask=(a_m < M) & (a_k < K),
                    other=0.0,
                    eviction_policy='evict_first')

        # B load: stride_bk=1, stride_bn=64  -> B[k, n] = arg_B + k*1 + n*64
        b_k = (offs_k + k_start)[:, None]
        b_n = rn[None, :]
        b = tl.load(arg_B + b_k * 1 + b_n * 64,
                    mask=(b_k < K) & (b_n < N),
                    other=0.0,
                    eviction_policy='evict_last')

        acc += tl.dot(a, b, allow_tf32=False, out_dtype=ACC_TYPE)

    # store output: stride_om=128, stride_on=1
    out_m = rm[:, None]
    out_n = rn[None, :]
    tl.store(out_ptr0 + out_m * 128 + out_n,
             acc,
             mask=(out_m < M) & (out_n < N))

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
