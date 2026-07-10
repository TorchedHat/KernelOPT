import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({}, num_warps=8,  num_stages=3),
    triton.Config({}, num_warps=8,  num_stages=4),
    triton.Config({}, num_warps=4,  num_stages=4),
    triton.Config({}, num_warps=16, num_stages=3),
    triton.Config({}, num_warps=8,  num_stages=5),
    triton.Config({}, num_warps=4,  num_stages=3),
], key=[])
@triton.jit
def triton_tem_fused_mm_t_0(arg_A, arg_B, out_ptr0):
    ACC_TYPE: tl.constexpr = tl.float32
    ALLOW_TF32: tl.constexpr = False
    BLOCK_M: tl.constexpr = 128
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 64
    GROUP_M: tl.constexpr = 8

    M = 1024
    N = 8192
    K = 8192

    if M * N == 0:
        return

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Compute base offsets for A and B tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Clean pointer arithmetic — no xindex broadcast, reduces register pressure
    a_ptrs = arg_A + offs_m[:, None] * K + offs_k[None, :]  # [BLOCK_M, BLOCK_K]
    b_ptrs = arg_B + offs_k[:, None] * N + offs_n[None, :]  # [BLOCK_K, BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k_idx * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    # Store result
    idx_m = offs_m[:, None]
    idx_n = offs_n[None, :]
    mask = (idx_m < M) & (idx_n < N)
    tl.store(out_ptr0 + idx_m * N + idx_n, acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (512, 1, 1)
    triton_tem_fused_mm_t_0[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8192, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
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
