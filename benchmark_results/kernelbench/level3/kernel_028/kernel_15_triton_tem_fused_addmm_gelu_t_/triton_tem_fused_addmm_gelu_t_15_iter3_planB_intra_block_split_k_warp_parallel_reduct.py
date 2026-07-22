import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def _split_k_gemm(
    arg_A, arg_B,
    out_ptr_ws,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(0)
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    mask_m = rm < M
    mask_n = rn < N
    k_start = pid_k * BLOCK_K
    rk = k_start + tl.arange(0, BLOCK_K)
    mask_k = rk < K
    a_ptrs = arg_A + rm[:, None] * K + rk[None, :]
    a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    b_ptrs = arg_B + rn[None, :] * K + rk[:, None]
    b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    a3 = tl.expand_dims(a, 2)
    b3 = tl.expand_dims(b, 0)
    partial = tl.sum(a3 * b3, axis=1)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    tl.atomic_add(out_ptr_ws + idx_m * N + idx_n, partial, mask=mask)


@triton.jit

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (1, 1, 1)
    _split_k_gemm[grid](arg_0, arg_1, arg_2, arg_3)
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
