import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4,  num_stages=3, maxnreg=128),
        triton.Config({}, num_warps=8,  num_stages=3, maxnreg=128),
        triton.Config({}, num_warps=4,  num_stages=2, maxnreg=128),
        triton.Config({}, num_warps=8,  num_stages=2, maxnreg=128),
        triton.Config({}, num_warps=4,  num_stages=4, maxnreg=128),
        triton.Config({}, num_warps=8,  num_stages=4, maxnreg=128),
        triton.Config({}, num_warps=4,  num_stages=1, maxnreg=128),
        triton.Config({}, num_warps=8,  num_stages=1, maxnreg=128),
        triton.Config({}, num_warps=2,  num_stages=3, maxnreg=128),
        triton.Config({}, num_warps=2,  num_stages=4, maxnreg=128),
        triton.Config({}, num_warps=4,  num_stages=3),
        triton.Config({}, num_warps=8,  num_stages=3),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused_convolution_6(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 501760
    N = 192
    K = 480
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 480
    stride_ak = 1
    stride_bk = 1
    stride_bn = 480

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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # A_ptr: load A as (BLOCK_M, BLOCK_K) with order=(1,0) for swizzled shared memory
    # A[m, k] = base + m*stride_am + k*stride_ak = base + m*480 + k*1
    # order=(1,0) means dim-1 (K) is contiguous — enables hardware swizzling
    # This eliminates 8-way shared memory bank conflicts from A tile loads
    A_ptr = tl.make_block_ptr(
        base=A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # B_ptr: load B transposed as (BLOCK_N, BLOCK_K) for coalesced access
    # B[k, n] = base + k*stride_bk + n*stride_bn = base + k*1 + n*480
    # Load as B_T[n, k] with shape=(N,K), strides=(stride_bn, stride_bk)=(480,1)
    # order=(1,0) means dim-1 (K) is contiguous — enables coalesced loads
    # This eliminates 8-way shared memory bank conflicts from strided N access
    B_ptr = tl.make_block_ptr(
        base=B,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a = tl.load(A_ptr, boundary_check=(0, 1))

        # Load B transposed: b_T shape is (BLOCK_N, BLOCK_K)
        # tl.trans() is a zero-cost metadata op giving (BLOCK_K, BLOCK_N) for tl.dot
        b_T = tl.load(B_ptr, boundary_check=(0, 1))
        b = tl.trans(b_T)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

        # Advance along K dimension
        A_ptr = tl.advance(A_ptr, (0, BLOCK_K))
        B_ptr = tl.advance(B_ptr, (0, BLOCK_K))


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 192*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (11760, 1, 1)
    triton_tem_fused_convolution_6[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 480, 224, 224), (24084480, 1, 107520, 480), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((192, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((501760, 192), (192, 1), device='cuda:0', dtype=torch.float32)
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
