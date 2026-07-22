import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused_add_addmm_native_layer_norm_t_view_11(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 65536
    N = 3072
    K = 768

    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # shape [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # shape [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                      # shape [BLOCK_K]

    # A: shape (M, K), row-major strides (K, 1)
    a_ptrs = arg_A + offs_m[:, None] * K + offs_k[None, :]  # [BLOCK_M, BLOCK_K]
    # B: physical shape (N=3072, K=768), strides (768, 1)
    # Accessed transposed as (K=768, N=3072): B[k, n] = arg_B + n*K + k*1
    b_ptrs = arg_B + offs_n[None, :] * K + offs_k[:, None] * 1  # [BLOCK_K, BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        # EVEN_K=True: K=768 divisible by BLOCK_K=64 — no mask needed
        a = tl.load(a_ptrs)   # [BLOCK_M, BLOCK_K]
        b = tl.load(b_ptrs)   # [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        a_ptrs += BLOCK_K     # advance K-dim of A (stride=1 per element)
        b_ptrs += BLOCK_K     # advance K-dim of B (stride=1 per element)

    # Store: M=65536 div by 64, N=3072 div by 64 — no mask needed
    out_ptrs = out_ptr0 + offs_m[:, None] * N + offs_n[None, :]  # [BLOCK_M, BLOCK_N]
    tl.store(out_ptrs, acc)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (12288, 1, 1)
    triton_tem_fused_add_addmm_native_layer_norm_t_view_11[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((65536, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
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
