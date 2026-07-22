import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4(arg_A, arg_B, out_ptr0):
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 128
    BLOCK_N: tl.constexpr = 32
    BLOCK_K: tl.constexpr = 32
    GROUP_M: tl.constexpr = 8
    ALLOW_TF32: tl.constexpr = False
    INDEX_DTYPE: tl.constexpr = tl.int32

    M = 125440
    N = 192
    K = 672

    if M * N == 0:
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

    # Create block pointers for A (shape: M x K, row-major strides: stride_am=672, stride_ak=1)
    A_ptr = tl.make_block_ptr(
        base=arg_A,
        shape=(M, K),
        strides=(672, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # Create block pointers for B (shape: K x N, strides: stride_bk=1, stride_bn=672)
    B_ptr = tl.make_block_ptr(
        base=arg_B,
        shape=(K, N),
        strides=(1, 672),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptr, boundary_check=(0, 1))
        b = tl.load(B_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        A_ptr = tl.advance(A_ptr, (0, BLOCK_K))
        B_ptr = tl.advance(B_ptr, (BLOCK_K, 0))

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 192 * idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (2940, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_4[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 672, 112, 112), (8429568, 1, 75264, 672), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((125440, 192), (192, 1), device='cuda:0', dtype=torch.float32)
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
