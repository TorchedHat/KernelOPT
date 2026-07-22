import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32

    M = 7840
    N = 40
    K = 240
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

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

    # Pre-compute TMA descriptors outside the k-loop
    # A has shape [M, K] = [7840, 240] with strides [240, 1] (row-major)
    # innermost stride is 1 (k-dim), so describe as [M, K] with strides [240, 1]
    a_desc = tl.make_tensor_descriptor(
        arg_A,
        shape=[M, K],
        strides=[240, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )

    # B has shape [K, N] = [240, 40] with strides [1, 240] (column-major)
    # stride_bk=1, stride_bn=240 means k is the contiguous dimension
    # TMA requires innermost stride=1, so describe B as [N, K] with strides [240, 1]
    # We load a [BLOCK_N, BLOCK_K] tile and transpose it to get [BLOCK_K, BLOCK_N]
    b_desc = tl.make_tensor_descriptor(
        arg_B,
        shape=[N, K],
        strides=[240, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        # Load A tile: [BLOCK_M, BLOCK_K]
        a = tl.load(a_desc, [pid_m * BLOCK_M, k_idx * BLOCK_K])

        # Load B tile as [BLOCK_N, BLOCK_K] then transpose to [BLOCK_K, BLOCK_N]
        b_t = tl.load(b_desc, [pid_n * BLOCK_N, k_idx * BLOCK_K])
        b = tl.trans(b_t)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 40 * idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')

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
