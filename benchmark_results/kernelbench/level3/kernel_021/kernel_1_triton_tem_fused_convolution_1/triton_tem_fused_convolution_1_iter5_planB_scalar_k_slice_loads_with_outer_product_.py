import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_convolution_1(in_ptr0, arg_A, arg_B, out_ptr0):
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 8
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 8388608
    N = 6
    K = 3
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3

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
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N

    a_m_mask = offs_a_m < M          # [BLOCK_M] boolean mask
    b_n_mask = offs_b_n < N          # [BLOCK_N] boolean mask

    # Load A[:, k] as 1-D vectors [BLOCK_M] -- stride_am=3, stride_ak=1
    a0 = tl.load(A + offs_a_m * stride_am + 0 * stride_ak, mask=a_m_mask, other=0.0)
    a1 = tl.load(A + offs_a_m * stride_am + 1 * stride_ak, mask=a_m_mask, other=0.0)
    a2 = tl.load(A + offs_a_m * stride_am + 2 * stride_ak, mask=a_m_mask, other=0.0)

    # Load B[k, :] as 1-D vectors [BLOCK_N] -- stride_bk=1, stride_bn=3
    b0 = tl.load(B + offs_b_n * stride_bn + 0 * stride_bk, mask=b_n_mask, other=0.0)
    b1 = tl.load(B + offs_b_n * stride_bn + 1 * stride_bk, mask=b_n_mask, other=0.0)
    b2 = tl.load(B + offs_b_n * stride_bn + 2 * stride_bk, mask=b_n_mask, other=0.0)

    # Outer-product FMA: straight-line sequence -> LLVM emits FFMA
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    acc += a0[:, None] * b0[None, :]   # k=0 contribution
    acc += a1[:, None] * b1[None, :]   # k=1 contribution
    acc += a2[:, None] * b2[None, :]   # k=2 contribution

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 6*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (65536, 1, 1)
    triton_tem_fused_convolution_1[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((6,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 3, 256, 256), (196608, 1, 768, 3), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((6, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8388608, 6), (6, 1), device='cuda:0', dtype=torch.float32)
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
