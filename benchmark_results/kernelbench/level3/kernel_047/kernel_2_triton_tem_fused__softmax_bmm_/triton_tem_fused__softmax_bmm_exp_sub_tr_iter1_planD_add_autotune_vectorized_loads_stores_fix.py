import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 25,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 100, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 25,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 50,  'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512, 'BLOCK_K': 25,  'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512, 'BLOCK_K': 100, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 50,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 20,  'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['BLOCK_M', 'BLOCK_N', 'BLOCK_K'],
)
@triton.jit
def triton_tem_fused__softmax_bmm_exp_sub_transpose_view_2(arg_A, arg_B, out_ptr0):
    M: tl.constexpr = 32
    N: tl.constexpr = 512
    K: tl.constexpr = 100
    BLOCK_M: tl.constexpr = 32   # overridden by autotune
    BLOCK_N: tl.constexpr = 128  # overridden by autotune
    BLOCK_K: tl.constexpr = 25   # overridden by autotune
    GROUP_M: tl.constexpr = 8    # overridden by autotune
    EVEN_K: tl.constexpr = (K % BLOCK_K == 0)
    ACC_TYPE: tl.constexpr = tl.float32
    ALLOW_TF32: tl.constexpr = False
    INDEX_DTYPE: tl.constexpr = tl.int32

    stride_aq = 3200
    stride_am = 1
    stride_ak = 32

    stride_bq = 51200
    stride_bk = 512
    stride_bn = 1

    pid = tl.program_id(0).to(INDEX_DTYPE)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM

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

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    A = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq)
    B = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            remaining = K - k * BLOCK_K
            a = tl.load(A, mask=rk[None, :] < remaining, other=0.)
            b = tl.load(B, mask=rk[:, None] < remaining, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    xindex = idx_n + N * idx_m + (N * M) * idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (4, 1, 1)
    triton_tem_fused__softmax_bmm_exp_sub_transpose_view_2[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((204800, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 100, 512), (51200, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 32, 512), (16384, 512, 1), device='cuda:0', dtype=torch.float32)
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
