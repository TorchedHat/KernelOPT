import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_t_view_0(arg_A, arg_B, out_ptr0):
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int32

    M : tl.constexpr = 65536
    N : tl.constexpr = 2304
    K : tl.constexpr = 768

    if M * N == 0:
        return

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

    # A: shape (M, K), strides (768, 1) — A[m, k] = base + m*768 + k
    a_block_ptr = tl.make_block_ptr(
        base=arg_A,
        shape=(M, K),
        strides=(768, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # B: logical B[k, n] = base + n + 2304*k → strides (2304, 1) for (K, N)
    # Verified: original xindex = idx_n + 2304*idx_m where idx_m=k_dim, idx_n=n_dim
    b_block_ptr = tl.make_block_ptr(
        base=arg_B,
        shape=(K, N),
        strides=(2304, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), eviction_policy='evict_first')
        b = tl.load(b_block_ptr, boundary_check=(0, 1), eviction_policy='evict_last')

        acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # Store output using block_ptr: out_ptr0 shape (M, N), strides (2304, 1)
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr0,
        shape=(M, N),
        strides=(2304, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(out_block_ptr, acc, boundary_check=(0, 1), eviction_policy='evict_first')

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
