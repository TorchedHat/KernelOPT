import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4(arg_A, arg_B, out_ptr0):
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ACC_TYPE : tl.constexpr = tl.float32
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    NUM_SMS : tl.constexpr = 132

    M = 512
    N = 512
    K = 96

    stride_aq = 49152
    stride_am = 96
    stride_ak = 1

    stride_bq = 49152
    stride_bk = 512
    stride_bn = 1

    grid_m = 8   # M // BLOCK_M = 512 // 64
    grid_n = 8   # N // BLOCK_N = 512 // 64
    tiles_per_batch = 64   # grid_m * grid_n
    total_tiles = 65536    # 1024 * 64

    sm_id = tl.program_id(0).to(INDEX_DTYPE)
    tile_id = sm_id

    while tile_id < total_tiles:
        idx_q = (tile_id // tiles_per_batch).to(INDEX_DTYPE)
        tile_mn = tile_id % tiles_per_batch

        # GROUP_M swizzle for L2 locality
        width = GROUP_M * grid_n  # = 64
        group_id = tile_mn // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = (group_id * GROUP_M + (tile_mn % group_size)).to(INDEX_DTYPE)
        pid_n = ((tile_mn % width) // group_size).to(INDEX_DTYPE)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        # stride_am=96, stride_ak=1: not row-major A, use plain modulo
        ram = rm % M
        # stride_bk=512, stride_bn=1: column-major B, use max_contiguous hint
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

        A_ptr = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq)
        B_ptr = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        # K=96, BLOCK_K=32 => exactly 3 iterations, EVEN_K=True
        for k in range(K, 0, -BLOCK_K):
            a = tl.load(A_ptr)
            b = tl.load(B_ptr)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_ptr += BLOCK_K * stride_ak
            B_ptr += BLOCK_K * stride_bk

        # rematerialize rm and rn to save registers
        rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm2[:, None]
        idx_n = rn2[None, :]
        mask = (idx_m < M) & (idx_n < N)

        # output stride layout: (262144, 512, 1)
        xindex = idx_n + 512 * idx_m + 262144 * idx_q
        tl.store(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask)

        tile_id += NUM_SMS

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (64, 1, 1)
    triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 8, 512, 96), (393216, 49152, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 8, 96, 512), (393216, 49152, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
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
