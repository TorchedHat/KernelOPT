import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_t_view_0(arg_A, arg_B, out_ptr0):
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 64
    GROUP_M: tl.constexpr = 8

    M = 65536
    N = 2304
    K = 768

    if M * N == 0:
        return

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(tl.int32)
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # A is [M, K] row-major, strides (K, 1) = (768, 1)
    a_block_ptr = tl.make_block_ptr(
        arg_A,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # B is stored as [N=2304, K=768] row-major with strides (768, 1).
    # We view it as [K, N] (transposed):
    #   moving +1 in K-dim costs stride 1 in memory
    #   moving +1 in N-dim costs stride K=768 in memory
    # order=(1,0): N-dimension (dim-1) is the fast/contiguous axis in shared memory,
    # matching how tl.dot reads B row-by-row, eliminating bank conflicts.
    b_block_ptr = tl.make_block_ptr(
        arg_B,
        shape=(K, N),
        strides=(1, K),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        acc += tl.dot(a, b, allow_tf32=False, out_dtype=ACC_TYPE)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 2304 * idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask=mask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (9216, 1, 1)
    triton_tem_fused_addmm_t_view_0[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((65536, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
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
