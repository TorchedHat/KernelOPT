import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_convolution_43(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1960
    N = 160
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

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

    # A is [M, K] row-major: shape=(M,K), strides=(stride_am=512, stride_ak=1)
    # K is contiguous (stride_ak=1), so order=(1,0) means dim1 (K) is fast in smem
    A_block_ptr = tl.make_block_ptr(
        base=A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # B is physically [N=160, K=512] row-major: B[n, k] = B_base + n*512 + k*1
    # Load as (BLOCK_N, BLOCK_K): rows=N (outer), cols=K (inner/contiguous)
    # Coalesced global loads (K contiguous) + N-major smem layout (no bank conflicts)
    # Define index vectors once before the loop
    rk_base = tl.arange(0, BLOCK_K)                         # shape (BLOCK_K,)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)            # shape (BLOCK_N,)

    # K-loop using block pointer advance with software pipelining (num_stages=3)
    for k_idx in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=3):
        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
        # Compute B tile using explicit pointer arithmetic
        k_offset = k_idx * BLOCK_K
        rk = k_offset + rk_base                              # shape (BLOCK_K,)
        b_ptrs = B + rn[:, None] * stride_bn + rk[None, :] * stride_bk  # (BLOCK_N, BLOCK_K)
        b_mask = (rn[:, None] < N) & (rk[None, :] < K)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)    # (BLOCK_N, BLOCK_K)
        # tl.trans gives (BLOCK_K, BLOCK_N) for tl.dot(a[BLOCK_M,BLOCK_K], b_T[BLOCK_K,BLOCK_N])
        acc += tl.dot(a, tl.trans(b_tile), allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 160*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (93, 1, 1)
    triton_tem_fused_convolution_43[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((160,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((160, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1960, 160), (160, 1), device='cuda:0', dtype=torch.float32)
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
