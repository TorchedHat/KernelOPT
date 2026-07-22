import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_convolution_1(in_ptr0, arg_A, arg_B, out_ptr0):
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 8
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 8388608
    N = 6
    K = 3
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3

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

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = rm
    offs_n = rn
    offs_a_m = rm % M
    offs_b_n = rn % N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Scalar outer-product FMA: K=3, fully unrolled, no shared memory, no bank conflicts
    for k in tl.static_range(3):
        a_col = tl.load(A + offs_a_m * stride_am + k * stride_ak,
                        mask=offs_m < M, other=0.0)  # shape [BLOCK_M]
        b_row = tl.load(B + k * stride_bk + offs_b_n * stride_bn,
                        mask=offs_n < N, other=0.0)  # shape [BLOCK_N]
        acc = acc + a_col[:, None] * b_row[None, :]  # outer product, pure register FMA

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 6*idx_m
    # 1D coalesced bias load + register broadcast (eliminates BLOCK_M-fold redundant loads)
    bias_1d = tl.load(in_ptr0 + rn, mask=rn < N, other=0.0)  # [BLOCK_N], 1 coalesced fetch
    tmp0 = bias_1d[None, :]  # register broadcast to [BLOCK_M, BLOCK_N], zero memory cost
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (65536, 1, 1)
    triton_tem_fused_convolution_1[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((6,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 3, 256, 256), (196608, 1, 768, 3), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((6, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8388608, 6), (6, 1), device='cuda:0', dtype=torch.float32)
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
