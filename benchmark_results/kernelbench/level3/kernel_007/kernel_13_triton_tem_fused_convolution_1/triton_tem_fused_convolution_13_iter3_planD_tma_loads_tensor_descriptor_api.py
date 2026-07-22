import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_convolution_13(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 7840
    N = 16
    K = 192
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 192
    stride_ak = 1
    stride_bk = 1
    stride_bn = 192

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

    # Create TMA tensor descriptors ONCE before the k-loop.
    # A: physical layout [M=7840, K=192], row-major strides [192, 1]
    # TMA requires last dim stride == 1 -> strides=[192, 1] satisfies this.
    a_desc = tl.make_tensor_descriptor(
        A, shape=[M, K], strides=[192, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    # B: physical layout is [N=16, K=192] with strides [192, 1]
    # (rand_strided((16,192,1,1),(192,1,1,1)) -> B[n, k] at offset n*192 + k*1)
    # TMA requires last dim stride == 1 -> strides=[192, 1] satisfies this.
    # We load [BLOCK_N, BLOCK_K] tiles and transpose to [BLOCK_K, BLOCK_N] for tl.dot.
    b_desc = tl.make_tensor_descriptor(
        B, shape=[N, K], strides=[192, 1], block_shape=[BLOCK_N, BLOCK_K]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        # TMA bulk async copy for A: loads [BLOCK_M, BLOCK_K] tile
        # boundary_check=(0,) because M=7840 is not a multiple of BLOCK_M=64
        a = tl.load(a_desc, [pid_m * BLOCK_M, k_idx * BLOCK_K], boundary_check=(0,))

        # TMA bulk async copy for B: loads [BLOCK_N, BLOCK_K] tile
        # N=16==BLOCK_N and K=192==6*BLOCK_K are exact, no boundary check needed
        b_t = tl.load(b_desc, [pid_n * BLOCK_N, k_idx * BLOCK_K])
        # Transpose [BLOCK_N, BLOCK_K] -> [BLOCK_K, BLOCK_N] for tl.dot
        b = tl.trans(b_t)

        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 16*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (123, 1, 1)
    triton_tem_fused_convolution_13[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((7840, 16), (16, 1), device='cuda:0', dtype=torch.float32)
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
