import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused_addmm_mm_relu_t_view_9(arg_A, arg_B, out_ptr0):
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 32
    BLOCK_N: tl.constexpr = 4
    BLOCK_K: tl.constexpr = 64
    GROUP_M: tl.constexpr = 8
    ALLOW_TF32: tl.constexpr = False
    INDEX_DTYPE: tl.constexpr = tl.int32
    NUM_SMS: tl.constexpr = 132

    A = arg_A
    B = arg_B

    M = 169
    N = 3
    K = 512

    if M * N == 0:
        return

    stride_am = 512  # K
    stride_ak = 1

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = grid_m * grid_n

    sm_id = tl.program_id(0).to(INDEX_DTYPE)

    for tile_id in tl.range(sm_id, total_tiles, NUM_SMS):
        # re-order program ID for better L2 performance
        width = GROUP_M * grid_n
        group_id = tile_id // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # offs_a_m: A has stride_am=512=K, stride_ak=1, M=169>=BLOCK_M, K>1 => use max_contiguous
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        # offs_b_n: simple modulo for N dimension
        offs_b_n = rn % N

        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

            # Load A tile: A[m, k] = A_ptr + m*512 + k
            idx_m = offs_a_m[:, None]
            idx_k = a_k_idx_vals
            xindex = idx_k + 512 * idx_m
            a = tl.load(A + xindex)

            # Load B tile: B has shape (3, 512) stride (512, 1)
            # B[n, k] = B_ptr + n*512 + k => index = k + 512*n
            idx_k2 = b_k_idx_vals          # [BLOCK_K, 1]
            idx_n = offs_b_n[None, :]      # [1, BLOCK_N]
            b = tl.load(B + (idx_k2 + 512 * idx_n))

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

        # Store result
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)
        xindex = idx_n + 3 * idx_m
        tl.store(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (6, 1, 1)
    triton_tem_fused_addmm_mm_relu_t_view_9[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((1, 13, 13, 512), (86528, 6656, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((3, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((169, 3), (3, 1), device='cuda:0', dtype=torch.float32)
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
