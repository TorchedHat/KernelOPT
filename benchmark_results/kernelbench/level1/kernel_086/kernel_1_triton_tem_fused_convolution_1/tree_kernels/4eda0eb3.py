import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_convolution_1(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int64
    A = arg_A
    B = arg_B

    M = 16777216
    N = 128
    K = 64
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 64
    stride_ak = 1
    stride_bk = 1
    stride_bn = 64

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

    # Cast offsets to int32 for tl.make_block_ptr (only supports 32-bit offsets)
    off_m = (pid_m * BLOCK_M).to(tl.int32)
    off_n = (pid_n * BLOCK_N).to(tl.int32)

    # Create TMA block pointers before the k-loop
    # A: shape (M, K), strides (stride_am=64, stride_ak=1), row-major K innermost
    A_block_ptr = tl.make_block_ptr(
        base=A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(off_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    # B: shape (K, N), strides (stride_bk=1, stride_bn=64), N innermost
    B_block_ptr = tl.make_block_ptr(
        base=B,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, off_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )
    # Output block pointer
    Out_block_ptr = tl.make_block_ptr(
        base=out_ptr0,
        shape=(M, N),
        strides=(N, 1),
        offsets=(off_m, off_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_block_ptr, boundary_check=(0, 1))
        b = tl.load(B_block_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))

    tl.store(Out_block_ptr, acc, boundary_check=(0, 1))

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (131072, 1, 1)
    triton_tem_fused_convolution_1[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((16, 64, 1024, 1024), (67108864, 1, 65536, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16777216, 128), (128, 1), device='cuda:0', dtype=torch.float32)
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
