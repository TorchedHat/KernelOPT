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
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused_convolution_56(in_ptr0, arg_A, arg_B, out_ptr0):
    BLOCK_M: tl.constexpr = 32
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 64
    GROUP_M: tl.constexpr = 8
    M = 1960
    N = 144
    K = 512
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    pid = tl.program_id(0).to(tl.int32)
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

    # TMA block pointer for A [M, K] row-major
    A_block_ptr = tl.make_block_ptr(
        base=arg_A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # TMA block pointer for B viewed as [K, N]:
    # B stored as [N=144, K=512] with strides (512, 1), so B[k,n] = base + n*512 + k*1
    # strides for [K, N] view: stride_bk=1 (along K), stride_bn=512 (along N)
    B_block_ptr = tl.make_block_ptr(
        base=arg_B,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_block_ptr, boundary_check=(0, 1))
        b = tl.load(B_block_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    tmp0 = tl.load(in_ptr0 + idx_n, mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + idx_n + N * idx_m, tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (93, 1, 1)
    triton_tem_fused_convolution_56[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((144,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((144, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1960, 144), (144, 1), device='cuda:0', dtype=torch.float32)
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
