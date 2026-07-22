import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_mean_relu_sigmoid_t_view_33(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 1
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 1000
    K = 1408
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1408
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1408

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

    # A: shape [M, K] with stride_am=1408, stride_ak=1 -> row-major
    # Use max_contiguous hint since M=2 <= BLOCK_M=2 and stride_am=K=1408
    offs_a_m = rm % M

    # B: shape [N, K] with stride_bn=1408, stride_bk=1
    # Use simple modulo to safely handle boundary tile (last tile: rn=896..1023, N=1000)
    offs_b_n = rn % N

    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 1408 * idx_m
        a = tl.load(A + xindex)

        # Load B tile directly as [BLOCK_K, BLOCK_N] — no transpose needed!
        # B[n, k] = B_ptr + n * 1408 + k * 1
        # We want b_tile[k, n] = B[offs_b_n[n], offs_k[k] + k_start]
        # = B_ptr + offs_b_n[n] * 1408 + (offs_k[k] + k_start) * 1
        k_start = k_idx * BLOCK_K
        b_tile = tl.load(B + (offs_k[:, None] + k_start) * 1 + offs_b_n[None, :] * 1408)
        # b_tile shape: [BLOCK_K, BLOCK_N] — K-outer (stride 1, contiguous), N-inner (stride 1408)
        # a: [BLOCK_M, BLOCK_K] x b_tile: [BLOCK_K, BLOCK_N] -> acc: [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, b_tile, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1000 * idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (32, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_mean_relu_sigmoid_t_view_33[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((1000,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2, 1408, 1, 1), (1408, 1, 2816, 2816), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1000, 1408), (1408, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
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
