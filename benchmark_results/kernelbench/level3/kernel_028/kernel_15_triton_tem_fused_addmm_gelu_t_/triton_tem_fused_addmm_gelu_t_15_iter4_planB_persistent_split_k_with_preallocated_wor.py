import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_gelu_t_15(in_ptr0, arg_A, arg_B, out_ptr0):
    # Split-K partial kernel: each of 32 blocks handles BLOCK_K_CHUNK=64 K-elements
    # out_ptr0 is used as the pre-zeroed FP32 workspace for atomic accumulation
    BLOCK_M: tl.constexpr = 2
    BLOCK_N: tl.constexpr = 16
    BLOCK_K_CHUNK: tl.constexpr = 64

    M = 2
    N = 10
    K = 2048

    sk = tl.program_id(0)  # 0..31
    k_start = sk * BLOCK_K_CHUNK

    rm = tl.arange(0, BLOCK_M)                        # [0, 1]
    rn = tl.arange(0, BLOCK_N)                        # [0..15]
    rk = k_start + tl.arange(0, BLOCK_K_CHUNK)        # [k_start..k_start+63]

    mask_m = rm < M
    mask_n = rn < N
    mask_k = rk < K

    # A: shape (2, 2048), row-major stride (2048, 1)
    a_ptrs = arg_A + rm[:, None] * K + rk[None, :]    # [BLOCK_M, BLOCK_K_CHUNK]
    a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

    # B: shape (10, 2048), row-major stride (2048, 1) => B^T[k, n]
    b_ptrs = arg_B + rn[None, :] * K + rk[:, None]    # [BLOCK_K_CHUNK, BLOCK_N]
    b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

    # Register-only 3D broadcast accumulation (no tl.dot, no shared memory)
    a3 = tl.expand_dims(a, 2)    # [BLOCK_M, BLOCK_K_CHUNK, 1]
    b3 = tl.expand_dims(b, 0)    # [1, BLOCK_K_CHUNK, BLOCK_N]
    partial = tl.sum(a3 * b3, axis=1)  # [BLOCK_M, BLOCK_N]

    # Atomic add into workspace (out_ptr0), stride N=10 actual
    mask = mask_m[:, None] & mask_n[None, :]
    tl.atomic_add(out_ptr0 + rm[:, None] * N + rn[None, :], partial, mask=mask)


@triton.jit

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (1, 1, 1)
    triton_tem_fused_addmm_gelu_t_15[grid](arg_0, arg_1, arg_2, arg_3)
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
