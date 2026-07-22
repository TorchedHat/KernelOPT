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
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=4),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused_addmm_native_layer_norm_t_view_1(arg_A, arg_B, out_ptr0):
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8

    M = 65536
    N = 2304
    K = 768

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

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Scalar offset index tensors (replaces tl.make_block_ptr)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                     # [BLOCK_K]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_idx * BLOCK_K
        # A: shape (M=65536, K=768), row-major strides (K=768, 1)
        a_idx = offs_m[:, None] * K + (k_start + offs_k[None, :])
        a_mask = (offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < K)
        a = tl.load(arg_A + a_idx, mask=a_mask, other=0.0, eviction_policy='evict_first')
        # B: physical shape (N=2304, K=768) row-major, viewed as (K=768, N=2304)
        # B[k, n] lives at flat offset n*K + k = offs_n*768 + offs_k
        b_idx = offs_n[None, :] * K + (k_start + offs_k[:, None])
        b_mask = ((k_start + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        b = tl.load(arg_B + b_idx, mask=b_mask, other=0.0, eviction_policy='evict_first')
        acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)

    # Store: out_ptr0 shape (M=65536, N=2304), row-major strides (N=2304, 1)
    out_idx = offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr0 + out_idx, acc, mask=out_mask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (9216, 1, 1)
    triton_tem_fused_addmm_native_layer_norm_t_view_1[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((65536, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
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
