import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def split_k_gemm_kernel(arg_A, arg_B, scratch_ptr,
                         M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr,
                         BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    k_id = tl.program_id(0)  # 0..SPLIT_K-1
    k_start = k_id * BLOCK_K
    k_end = tl.minimum(k_start + BLOCK_K, K)

    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    mask_m = rm < M
    mask_n = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Sub-tile size 32 to reduce register pressure
    for k_off in range(0, BLOCK_K, 32):
        rk = k_start + k_off + tl.arange(0, 32)
        mask_k = rk < k_end

        # A: shape (M, K), row-major stride (K, 1)
        a = tl.load(arg_A + rm[:, None] * K + rk[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # B: shape (N, K), row-major stride (K, 1) — B^T[k, n]
        b = tl.load(arg_B + rn[None, :] * K + rk[:, None],
                    mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # a: [BLOCK_M, 32] -> [BLOCK_M, 32, 1]
        # b: [32, BLOCK_N] -> [1, 32, BLOCK_N]
        a3 = tl.expand_dims(a, 2)
        b3 = tl.expand_dims(b, 0)
        acc += tl.sum(a3 * b3, axis=1)  # [BLOCK_M, BLOCK_N]

    # Write to exclusive scratch slice — no atomics, no races
    off = k_id * M * N + rm[:, None] * N + rn[None, :]
    tl.store(scratch_ptr + off, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (1, 1, 1)
    split_k_gemm_kernel[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((10,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2, 10), (10, 1), device='cuda:0', dtype=torch.float32)
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
