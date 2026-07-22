import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=4),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_mean_relu_sigmoid_t_view_33(in_ptr0, arg_A, arg_B, out_ptr0):
    # Fixed problem dimensions
    M: tl.constexpr = 2
    N: tl.constexpr = 1000
    K: tl.constexpr = 1408
    BLOCK_M: tl.constexpr = 2
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 64
    GROUP_M: tl.constexpr = 1

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # TMA block pointers for A: shape (M=2, K=1408), strides (1408, 1), row-major
    A_ptr = tl.make_block_ptr(
        base=arg_A,
        shape=(M, K),
        strides=(1408, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # TMA block pointers for B: logical shape (K=1408, N=1000), strides (1, 1408)
    # arg_B has physical shape (1000, 1408) with strides (1408, 1),
    # but we treat it as B[k, n] where stride_k=1, stride_n=1408
    B_ptr = tl.make_block_ptr(
        base=arg_B,
        shape=(K, N),
        strides=(1, 1408),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptr, boundary_check=(0, 1), padding_option='zero')
        b = tl.load(B_ptr, boundary_check=(0, 1), padding_option='zero')
        acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
        A_ptr = tl.advance(A_ptr, (0, BLOCK_K))
        B_ptr = tl.advance(B_ptr, (BLOCK_K, 0))

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    # load bias (shape [1000]) and broadcast across BLOCK_M rows
    bias = tl.load(in_ptr0 + rn[None, :], mask=rn[None, :] < N, other=0.0, eviction_policy='evict_last')
    tmp1 = acc + bias
    tl.store(out_ptr0 + rm[:, None] * N + rn[None, :], tmp1, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (32, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_mean_relu_sigmoid_t_view_33[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((1000,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2, 1408, 1, 1), (1408, 1, 2816, 2816), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1000, 1408), (1408, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
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
