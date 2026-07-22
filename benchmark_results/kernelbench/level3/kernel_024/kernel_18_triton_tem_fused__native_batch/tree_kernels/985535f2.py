import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=4),
    triton.Config({}, num_warps=8, num_stages=4),
], key=[])
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_18(arg_A, arg_B, out_ptr1):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 1
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 216
    K = 864
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 864
    stride_ak = 1
    stride_bk = 1
    stride_bn = 864

    pid = tl.program_id(0).to(INDEX_DTYPE)
    num_programs = tl.num_programs(0)

    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n

    for tile_id in tl.range(pid, total_tiles, num_programs):
        pid_m = (tile_id // num_tiles_n).to(INDEX_DTYPE)
        pid_n = (tile_id % num_tiles_n).to(INDEX_DTYPE)

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # No modulo guard for rm — use explicit masking on A load instead
        offs_b_n = rn % N

        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

            # A load: mask padded rows (rows M..BLOCK_M-1) to 0.0
            idx_m_a = rm[:, None]
            idx_n_a = a_k_idx_vals
            xindex_a = idx_n_a + 864 * idx_m_a
            a_mask = (rm[:, None] < M) & (a_k_idx_vals < K)
            a = tl.load(A + xindex_a, mask=a_mask, other=0.0)

            idx_m_b = b_k_idx_vals
            idx_n_b = offs_b_n[None, :]
            xindex_b = idx_n_b + 216 * idx_m_b
            b_mask = (b_k_idx_vals < K) & (offs_b_n[None, :] < N)
            b = tl.load(B + (tl.broadcast_to(idx_m_b + 864 * idx_n_b, [BLOCK_K, BLOCK_N])).broadcast_to(xindex_b.shape), mask=b_mask, other=0.0)

            # Use tl.dot for MMA hardware acceleration
            acc += tl.dot(a, b, allow_tf32=False)

        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)

        # inductor generates a suffix
        xindex = idx_n + 216 * idx_m
        tmp0 = tl.full([1], 0, tl.int32)
        tmp1 = tl.maximum(tmp0, acc)
        tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (7, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_18[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 864, 1, 1), (864, 1, 864, 864), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((216, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 216, 1, 1), (216, 1, 216, 216), device='cuda:0', dtype=torch.float32)
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
