import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_7(arg_A, arg_B, out_ptr1):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 24
    K = 96
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 96
    stride_ak = 1
    stride_bk = 1
    stride_bn = 96

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
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        
        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 96*idx_m
        a = tl.load(A + (xindex))

        # Rewritten B load: access B[k,n] = B_ptr + k*1 + n*96 with stride-1 along K
        # (eliminates 17-way bank conflict from the original stride-96-N access pattern)
        idx_k = offs_k[:, None] + (k_idx * BLOCK_K)   # shape [BLOCK_K, 1]
        idx_n = offs_b_n[None, :]                       # shape [1, BLOCK_N]
        b = tl.load(B + (idx_k + 96 * idx_n))           # shape [BLOCK_K, BLOCK_N]

        # Manual multiply-accumulate: a is [BLOCK_M, BLOCK_K], b is [BLOCK_K, BLOCK_N]
        # acc[m, n] += sum_k a[m, k] * b[k, n]
        # Use tl.dot with padded tiles to satisfy minimum dimension requirement (>=16)
        # Pad BLOCK_M dimension to 16 for tl.dot compatibility
        a_pad = tl.reshape(a, [1, BLOCK_M, BLOCK_K])
        b_t = tl.trans(b)  # [BLOCK_N, BLOCK_K]
        # Compute outer product accumulation via broadcasting
        # a: [BLOCK_M, BLOCK_K, 1], b: [1, BLOCK_K, BLOCK_N]
        a_exp = tl.expand_dims(a, 2)   # [BLOCK_M, BLOCK_K, 1]
        b_exp = tl.expand_dims(b, 0)   # [1, BLOCK_K, BLOCK_N]
        acc += tl.sum(a_exp * b_exp, axis=1)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 24*idx_m
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tl.maximum(tmp0, acc)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (1, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_7[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 96, 1, 1), (96, 1, 96, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 24, 1, 1), (24, 1, 24, 24), device='cuda:0', dtype=torch.float32)
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
