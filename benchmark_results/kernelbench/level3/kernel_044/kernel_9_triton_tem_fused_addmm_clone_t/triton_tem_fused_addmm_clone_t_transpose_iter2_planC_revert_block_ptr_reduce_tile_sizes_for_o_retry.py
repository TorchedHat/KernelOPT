import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_clone_t_transpose_view_9(arg_A, arg_B, out_ptr0):
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 32
    GROUP_M: tl.constexpr = 8
    INDEX_DTYPE: tl.constexpr = tl.int32

    M = 65536
    N = 768
    K = 768

    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768
    stride_cm = 768
    stride_cn = 1

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

    # Scalar offset arithmetic
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize pointers
    # A is (M, K) with stride_am=768, stride_ak=1
    A_ptrs = arg_A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B is (K, N) with stride_bk=1, stride_bn=768
    B_ptrs = arg_B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] + k * BLOCK_K < K),
                    other=0.0)
        b = tl.load(B_ptrs,
                    mask=(offs_k[:, None] + k * BLOCK_K < K) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Store result
    C_ptrs = out_ptr0 + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (3072, 1, 1)
    triton_tem_fused_addmm_clone_t_transpose_view_9[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 512, 8, 96), (393216, 768, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((65536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
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
