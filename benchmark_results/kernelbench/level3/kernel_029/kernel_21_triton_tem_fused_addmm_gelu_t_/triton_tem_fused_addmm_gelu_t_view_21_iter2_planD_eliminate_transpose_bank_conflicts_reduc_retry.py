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
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused_addmm_gelu_t_view_21(arg_A, arg_B, out_ptr0):
    ALLOW_TF32: tl.constexpr = False
    ACC_TYPE: tl.constexpr = tl.float32
    INDEX_DTYPE: tl.constexpr = tl.int32
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 64
    GROUP_M: tl.constexpr = 8

    M = 7840
    N = 192
    K = 768

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

    # Create block pointers for A [M=7840, K=768] row-major
    a_block_ptr = tl.make_block_ptr(
        base=arg_A,
        shape=(M, K),
        strides=(768, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # B is stored as [N=192, K=768] row-major (stride 768, 1)
    # B[n, k] is at offset n*768 + k
    # We need B^T: B^T[k, n] = B[n, k] = base[n*768 + k] = base[k*1 + n*768]
    # So treat B as shape (K, N) with strides (1, 768):
    #   element [k, n] -> offset k*1 + n*768 = n*768 + k = B[n, k] ✓
    b_block_ptr = tl.make_block_ptr(
        base=arg_B,
        shape=(K, N),
        strides=(1, 768),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')  # already [BLOCK_K, BLOCK_N], no trans needed

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    xindex = idx_n + 192 * idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (246, 1, 1)
    triton_tem_fused_addmm_gelu_t_view_21[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((7840, 192), (192, 1), device='cuda:0', dtype=torch.float32)
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
