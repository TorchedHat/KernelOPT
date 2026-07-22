import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_gelu_t_view_5(arg_A, arg_B, out_ptr0):
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8

    M = 31360
    N = 96
    K = 384
    if M * N == 0:
        return
    stride_am = 384
    stride_ak = 1
    # B physical shape (96,384) strides (384,1) => read as (K=384, N=96)
    # stride_bk=1 (stride along K dim), stride_bn=384 (stride along N dim)
    stride_bk = 1
    stride_bn = 384

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # TMA block pointers - address generation offloaded to Hopper TMA hardware
    A_block_ptr = tl.make_block_ptr(
        base=arg_A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # B physical shape (96,384) strides (384,1) => read as (K=384, N=96) strides (1, 384)
    B_block_ptr = tl.make_block_ptr(
        base=arg_B,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_block_ptr, boundary_check=(0, 1))
        b = tl.load(B_block_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptr0 + rm[:, None] * N + rn[None, :], acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (245, 1, 1)
    triton_tem_fused_addmm_gelu_t_view_5[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 3136, 384), (1204224, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 96), (96, 1), device='cuda:0', dtype=torch.float32)
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
