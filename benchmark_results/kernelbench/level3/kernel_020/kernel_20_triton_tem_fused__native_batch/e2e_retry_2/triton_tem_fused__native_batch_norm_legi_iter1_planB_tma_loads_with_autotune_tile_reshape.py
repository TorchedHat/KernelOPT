import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=4),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=4),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
], key=[])
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20(arg_A, arg_B, out_ptr0):
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int32

    M = 7840
    N = 40
    K = 240

    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
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

    # TMA-style block pointer for A: shape (M, K), strides (240, 1)
    a_block_ptr = tl.make_block_ptr(
        base=arg_A,
        shape=(M, K),
        strides=(240, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # TMA-style block pointer for B: shape (K, N), strides (1, 240)
    # arg_B has shape (40,240,1,1) stride (240,1,1,1)
    # In [K,N] view: K=240 rows with stride 1, N=40 cols with stride 240
    b_block_ptr = tl.make_block_ptr(
        base=arg_B,
        shape=(K, N),
        strides=(1, 240),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # Store output: shape (M, N), strides (40, 1)
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr0,
        shape=(M, N),
        strides=(40, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(out_block_ptr, acc, boundary_check=(0, 1))

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (123, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((7840, 40), (40, 1), device='cuda:0', dtype=torch.float32)
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
