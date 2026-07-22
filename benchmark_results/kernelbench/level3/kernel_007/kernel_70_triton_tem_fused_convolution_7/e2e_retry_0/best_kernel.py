import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_convolution_70(in_ptr0, arg_A, arg_B, out_ptr0):
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    M : tl.constexpr = 1960
    N : tl.constexpr = 128
    K : tl.constexpr = 528

    if M * N == 0:
        return

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # TMA block pointers — contiguous tile access, eliminates bank conflicts
    # A: shape (M=1960, K=528), strides (528, 1) — row-major
    A_ptr = tl.make_block_ptr(
        base=arg_A,
        shape=(M, K),
        strides=(528, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # B: original tensor shape (128, 528) with strides (528, 1),
    # viewed as (K=528, N=128) transposed: stride_bk=1, stride_bn=528
    B_ptr = tl.make_block_ptr(
        base=arg_B,
        shape=(K, N),
        strides=(1, 528),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptr, boundary_check=(0, 1), padding_option='zero')
        b = tl.load(B_ptr, boundary_check=(0, 1), padding_option='zero')
        acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
        A_ptr = tl.advance(A_ptr, (0, BLOCK_K))
        B_ptr = tl.advance(B_ptr, (BLOCK_K, 0))

    # epilogue: bias add and store
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    bias = tl.load(in_ptr0 + idx_n, mask=(idx_n < N), other=0.0, eviction_policy='evict_last')
    tmp1 = acc + bias
    tl.store(out_ptr0 + idx_n + N * idx_m, tmp1, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (124, 1, 1)
    triton_tem_fused_convolution_70[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 528, 14, 14), (103488, 1, 7392, 528), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 528, 1, 1), (528, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1960, 128), (128, 1), device='cuda:0', dtype=torch.float32)
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
