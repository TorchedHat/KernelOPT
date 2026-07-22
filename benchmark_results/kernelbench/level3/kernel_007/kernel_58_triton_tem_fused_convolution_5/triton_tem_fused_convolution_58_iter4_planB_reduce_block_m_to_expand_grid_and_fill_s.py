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
        triton.Config({}, num_warps=2, num_stages=5),
        triton.Config({}, num_warps=4, num_stages=5),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused_convolution_58(in_ptr0, arg_A, arg_B, out_ptr0):
    ALLOW_TF32: tl.constexpr = False
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 16
    BLOCK_N: tl.constexpr = 32
    BLOCK_K: tl.constexpr = 32
    GROUP_M: tl.constexpr = 8

    M = 1960
    N = 32
    K = 512

    if M * N == 0:
        return

    pid = tl.program_id(0)
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_idx * BLOCK_K
        # A is row-major with strides (512, 1)
        a_ptrs = arg_A + offs_m[:, None] * 512 + (k_start + offs_k[None, :]) * 1
        mask_a = (offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        # B is column-major with strides (1, 512)
        b_ptrs = arg_B + (k_start + offs_k[:, None]) * 1 + offs_n[None, :] * 512
        mask_b = ((k_start + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    tmp0 = tl.load(in_ptr0 + idx_n, mask=(idx_n < N), eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (idx_n + N * idx_m), tmp1, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (62, 1, 1)
    triton_tem_fused_convolution_58[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1960, 32), (32, 1), device='cuda:0', dtype=torch.float32)
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
