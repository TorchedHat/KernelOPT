import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_gelu_t_15(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 256
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32

    M = 2
    N = 10
    K = 2048

    if M * N == 0:
        return

    pid = tl.program_id(0)

    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    mask_m = rm < M
    mask_n = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        mask_k = rk < K

        a_ptrs = arg_A + rm[:, None] * 2048 + rk[None, :]
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        b_ptrs = arg_B + rn[None, :] * 2048 + rk[:, None]
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        a3 = tl.expand_dims(a, 2)
        b3 = tl.expand_dims(b, 0)
        acc += tl.sum(a3 * b3, axis=1)

    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = mask_m[:, None] & mask_n[None, :]

    xindex = idx_n + 10 * idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)


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
