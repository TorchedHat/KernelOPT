import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=2),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused_addmm_relu_t_8(in_ptr0, arg_A, arg_B, out_ptr0):
    ACC_TYPE: tl.constexpr = tl.float32
    ALLOW_TF32: tl.constexpr = False
    INDEX_DTYPE: tl.constexpr = tl.int32
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 32
    BLOCK_K: tl.constexpr = 32
    GROUP_M: tl.constexpr = 8

    M = 4096
    N = 20
    K = 84
    stride_am = 84
    stride_ak = 1
    stride_bk = 1
    stride_bn = 84

    if M * N == 0:
        return

    # based on triton.ops.matmul
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # TMA-style block pointer for A: shape (M, K), row-major
    A_ptr = tl.make_block_ptr(
        base=arg_A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # TMA-style block pointer for B: shape (K, N)
    # B is stored as (N, K) with stride_bn=84, stride_bk=1
    # viewed as (K, N): strides (stride_bk=1, stride_bn=84)
    B_ptr = tl.make_block_ptr(
        base=arg_B,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptr, boundary_check=(0, 1), padding_option='zero')
        b = tl.load(B_ptr, boundary_check=(0, 1), padding_option='zero')
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        A_ptr = tl.advance(A_ptr, (0, BLOCK_K))
        B_ptr = tl.advance(B_ptr, (BLOCK_K, 0))

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 20 * idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (128, 1, 1)
    triton_tem_fused_addmm_relu_t_8[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((20,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4096, 84), (84, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((20, 84), (84, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((4096, 20), (20, 1), device='cuda:0', dtype=torch.float32)
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
