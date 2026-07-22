import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_convolution_t_view_3(in_ptr0, arg_A, arg_B, out_ptr0):
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    SPLIT_K : tl.constexpr = 32
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32

    M : tl.constexpr = 10
    N : tl.constexpr = 128
    K : tl.constexpr = 8192

    if M * N == 0:
        return

    # Program IDs
    pid   = tl.program_id(0)
    pid_k = tl.program_id(2)  # which K-slice this CTA owns

    # Tile grid dimensions
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # L2-friendly re-ordering of output tiles
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Row/col ranges for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Each pid_k handles a strided slice of K.
    # Iteration k_idx covers k_offset = pid_k * BLOCK_K + k_idx * SPLIT_K * BLOCK_K
    k_iters = tl.cdiv(K, SPLIT_K * BLOCK_K)
    for k_idx in range(0, k_iters):
        k_offset = pid_k * BLOCK_K + k_idx * SPLIT_K * BLOCK_K
        k_range = k_offset + tl.arange(0, BLOCK_K)

        # Load A tile: shape (BLOCK_M, BLOCK_K)
        # A = arg_A: (10, 8192) with stride_am=8192, stride_ak=1
        idx_am = rm[:, None]       # (BLOCK_M, 1)
        idx_ak = k_range[None, :]  # (1, BLOCK_K)
        mask_a = (idx_am < M) & (idx_ak < K)
        a = tl.load(arg_A + idx_am * 8192 + idx_ak, mask=mask_a, other=0.0)

        # Load B tile: shape (BLOCK_K, BLOCK_N)
        # B = arg_B: (128, 8192) with strides (8192, 1)
        # B[n, k] at ptr + n*8192 + k*1
        # We want b[k, n] = B[n, k] → load ptr + rn*8192 + k_range
        idx_bk = k_range[:, None]  # (BLOCK_K, 1)
        idx_bn = rn[None, :]       # (1, BLOCK_N)
        mask_b = (idx_bk < K) & (idx_bn < N)
        b = tl.load(arg_B + idx_bn * 8192 + idx_bk, mask=mask_b, other=0.0)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # Write partial results via atomic_add (out_ptr0 was zeroed before launch)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    xindex = idx_n + 128 * idx_m
    tl.atomic_add(out_ptr0 + xindex, acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (4, 1, 1)
    triton_tem_fused_addmm_convolution_t_view_3[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 128, 8, 8), (8192, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 128), (128, 1), device='cuda:0', dtype=torch.float32)
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
