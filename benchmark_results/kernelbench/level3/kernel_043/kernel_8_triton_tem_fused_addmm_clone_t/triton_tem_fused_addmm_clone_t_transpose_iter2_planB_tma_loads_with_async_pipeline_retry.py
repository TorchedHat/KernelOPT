import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_clone_t_transpose_view_8(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    NUM_STAGES : tl.constexpr = 3
    A = arg_A
    B = arg_B

    M = 65536
    N = 768
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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

    # Build TMA descriptor for A: shape [M, K] = [65536, 768], row-major strides [768, 1]
    # A[m, k] at offset m*768 + k*1 — standard row-major
    desc_a = tl.make_tensor_descriptor(
        arg_A,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )

    # Build TMA descriptor for B: B is stored with stride_bk=1, stride_bn=768
    # B[k, n] at offset k*1 + n*768 — this is column-major for (K,N), or equivalently
    # B is stored as (N, K) row-major: B_as_NT[n, k] at offset n*768 + k*1
    # So we describe it as shape [N, K] with strides [768, 1] and load [BLOCK_N, BLOCK_K] tiles
    # then transpose to get [BLOCK_K, BLOCK_N] for tl.dot
    desc_b = tl.make_tensor_descriptor(
        arg_B,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K],
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        # Load A tile: [BLOCK_M, BLOCK_K] at [pid_m*BLOCK_M, k_idx*BLOCK_K]
        a = desc_a._load([pid_m * BLOCK_M, k_idx * BLOCK_K])

        # Load B tile as [BLOCK_N, BLOCK_K] from the (N, K) descriptor
        # at [pid_n*BLOCK_N, k_idx*BLOCK_K], then transpose to [BLOCK_K, BLOCK_N]
        b_nt = desc_b._load([pid_n * BLOCK_N, k_idx * BLOCK_K])
        b = tl.trans(b_nt)

        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 768*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (3072, 1, 1)
    triton_tem_fused_addmm_clone_t_transpose_view_8[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 512, 8, 96), (393216, 768, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((65536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
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
