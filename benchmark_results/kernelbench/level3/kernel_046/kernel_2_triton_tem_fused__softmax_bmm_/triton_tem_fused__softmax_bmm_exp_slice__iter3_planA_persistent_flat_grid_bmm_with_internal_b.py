import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__softmax_bmm_exp_slice_sub_transpose_view_2(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32

    M = 32
    N = 512
    K = 100

    stride_aq = 4800
    stride_am = 1
    stride_ak = 48

    stride_bq = 51200
    stride_bk = 512
    stride_bn = 1

    # Persistent kernel: flat tile_id over all (batch, m, n) tiles
    # total_tiles = 2048 * 1 * 4 = 8192
    # grid_m = 1, grid_n = 4, tiles_per_batch = 4
    total_tiles = 8192
    grid_m = 1
    grid_n = 4
    tiles_per_batch = 4  # grid_m * grid_n

    worker_id = tl.program_id(0).to(tl.int32)
    num_workers = tl.num_programs(0).to(tl.int32)
    tile_id = worker_id

    while tile_id < total_tiles:
        # Decode flat tile_id into (idx_q, pid_m, pid_n)
        idx_q = tile_id // tiles_per_batch
        local = tile_id % tiles_per_batch
        pid_m = local // grid_n
        pid_n = local % grid_n

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # vectorized load hints: signal 128-bit alignment to compiler
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

        rk = tl.arange(0, BLOCK_K)

        A = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq)
        B = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for k in range(K, 0, -BLOCK_K):
            if EVEN_K:
                a = tl.load(A)
                b = tl.load(B)
            else:
                a = tl.load(A, mask=rk[None, :] < k, other=0.)
                b = tl.load(B, mask=rk[:, None] < k, other=0.)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk

        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)

        # inductor generates a suffix
        xindex = idx_n + 512 * idx_m + 16384 * idx_q
        tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

        tile_id += num_workers

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (4, 1, 1)
    triton_tem_fused__softmax_bmm_exp_slice_sub_transpose_view_2[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((204800, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 100, 512), (51200, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 32, 512), (16384, 512, 1), device='cuda:0', dtype=torch.float32)
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
