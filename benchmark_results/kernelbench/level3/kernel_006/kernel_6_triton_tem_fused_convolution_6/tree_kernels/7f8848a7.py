import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=8),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused_convolution_6(in_ptr0, arg_A, arg_B, out_ptr0):
    BLOCK_M: tl.constexpr = 128
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 32
    GROUP_M: tl.constexpr = 8

    M: tl.constexpr = 501760
    N: tl.constexpr = 192
    K: tl.constexpr = 480

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # A: [M, K] row-major, strides=(480, 1), order=(1,0) means K-dim is contiguous
    a_block_ptr = tl.make_block_ptr(
        base=arg_A,
        shape=(M, K),
        strides=(480, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0))

    # B: [K, N] with strides=(1, 480), order=(0,1) means K-dim (dim-0) is contiguous
    # B[k, n] = arg_B + k*1 + n*480
    b_block_ptr = tl.make_block_ptr(
        base=arg_B,
        shape=(K, N),
        strides=(1, 480),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # epilogue: bias add + store
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    xindex = idx_n + 192 * idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (11760, 1, 1)
    triton_tem_fused_convolution_6[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 480, 224, 224), (24084480, 1, 107520, 480), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((192, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((501760, 192), (192, 1), device='cuda:0', dtype=torch.float32)
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
