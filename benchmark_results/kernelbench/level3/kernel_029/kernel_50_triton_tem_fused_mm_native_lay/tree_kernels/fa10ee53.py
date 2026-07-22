import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_mm_native_layer_norm_t_view_50(arg_A, arg_B, out_ptr0):
    ALLOW_TF32: tl.constexpr = False
    ACC_TYPE: tl.constexpr = tl.float32
    GROUP_M: tl.constexpr = 8
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 32
    M: tl.constexpr = 490
    N: tl.constexpr = 768
    K: tl.constexpr = 1536

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

    # A[m, k] with strides (1536, 1): base + m*1536 + k*1
    a_ptrs = arg_A + offs_m[:, None] * 1536 + offs_k[None, :]
    # B[k, n] with strides (1, 1536): base + k*1 + n*1536
    b_ptrs = arg_B + offs_k[:, None] * 1 + offs_n[None, :] * 1536

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(tl.cdiv(K, BLOCK_K)):
        k_offset = k_idx * BLOCK_K
        a_mask = (offs_m[:, None] < M) & ((offs_k[None, :] + k_offset) < K)
        b_mask = ((offs_k[:, None] + k_offset) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr0 + offs_m[:, None] * 768 + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (96, 1, 1)
    triton_tem_fused_mm_native_layer_norm_t_view_50[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 49, 1536), (75264, 1536, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 768), (768, 1), device='cuda:0', dtype=torch.float32)
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
