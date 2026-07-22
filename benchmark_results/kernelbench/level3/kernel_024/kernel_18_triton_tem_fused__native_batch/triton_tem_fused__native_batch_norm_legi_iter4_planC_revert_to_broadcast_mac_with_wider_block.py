import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({}, num_warps=1, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=1, num_stages=3),
    triton.Config({}, num_warps=2, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=2, num_stages=4),
    triton.Config({}, num_warps=4, num_stages=4),
], key=[])
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_18(arg_A, arg_B, out_ptr1):
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 32
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 216
    K = 864
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    # total_tiles = ceil(2/2) * ceil(216/256) = 1*1 = 1, no persistent loop needed
    pid_m = 0
    pid_n = 0

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k_idx * BLOCK_K

        # A: shape [BLOCK_M, BLOCK_K]
        a_k_vals = offs_k[None, :] + k_off
        a_mask = (rm[:, None] < M) & (a_k_vals < K)
        xindex_a = a_k_vals + 864 * rm[:, None]
        a = tl.load(A + xindex_a, mask=a_mask, other=0.0)

        # B: shape [BLOCK_K, BLOCK_N]
        b_k = offs_k + k_off  # shape [BLOCK_K]
        b_idx = rn[None, :] + 216 * b_k[:, None]  # shape [BLOCK_K, BLOCK_N]
        b_mask = (b_k[:, None] < K) & (rn[None, :] < N)
        b = tl.load(B + b_idx, mask=b_mask, other=0.0)

        # Broadcast MAC: avoids SMEM bank conflicts from tl.dot
        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N]
        # a[:, :, None] * b[None, :, :] -> [BLOCK_M, BLOCK_K, BLOCK_N]
        # sum over axis=1 (K) -> [BLOCK_M, BLOCK_N]
        acc += tl.sum(a[:, :, None] * b[None, :, :], axis=1)

    # epilogue
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
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
