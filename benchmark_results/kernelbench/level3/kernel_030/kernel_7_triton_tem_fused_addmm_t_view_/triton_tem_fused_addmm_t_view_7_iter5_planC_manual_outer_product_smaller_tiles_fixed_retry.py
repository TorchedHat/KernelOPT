import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_t_view_7(arg_A, arg_B, out_ptr0):
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 8
    BLOCK_N : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 169
    N = 512
    K = 2
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 2
    stride_ak = 1
    stride_bk = 1
    stride_bn = 2

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

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_a_m = rm % M
    offs_b_n = rn % N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Manual outer-product loop: range(2) with literal 2 — unrolled at JIT compile time
    # A layout: xindex = k*stride_ak + offs_a_m*stride_am = k*1 + offs_a_m*2
    # B layout: xindex = k*stride_bk + offs_b_n*stride_bn = k*1 + offs_b_n*2
    for k in range(2):
        a_col = tl.load(A + (k * stride_ak + offs_a_m * stride_am))  # [BLOCK_M]
        b_row = tl.load(B + (k * stride_bk + offs_b_n * stride_bn))  # [BLOCK_N]
        acc = acc + a_col[:, None].to(ACC_TYPE) * b_row[None, :].to(ACC_TYPE)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512 * idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (96, 1, 1)
    triton_tem_fused_addmm_t_view_7[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((1, 13, 13, 2), (338, 26, 2, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 2), (2, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((169, 512), (512, 1), device='cuda:0', dtype=torch.float32)
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
