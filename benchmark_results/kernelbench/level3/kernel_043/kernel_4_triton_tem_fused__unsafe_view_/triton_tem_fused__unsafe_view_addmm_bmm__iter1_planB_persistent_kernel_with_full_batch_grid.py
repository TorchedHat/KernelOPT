import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=4),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=4),
], key=[])
@triton.jit
def triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4(arg_A, arg_B, out_ptr0):
    # All constants hardcoded
    M: tl.constexpr = 512
    N: tl.constexpr = 512
    K: tl.constexpr = 96
    NUM_BATCHES: tl.constexpr = 1024
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 32
    NUM_SMS: tl.constexpr = 132

    stride_aq: tl.constexpr = 49152
    stride_am: tl.constexpr = 96
    stride_ak: tl.constexpr = 1

    stride_bq: tl.constexpr = 49152
    stride_bk: tl.constexpr = 512
    stride_bn: tl.constexpr = 1

    grid_m: tl.constexpr = M // BLOCK_M   # 8
    grid_n: tl.constexpr = N // BLOCK_N   # 8
    tiles_per_batch: tl.constexpr = grid_m * grid_n  # 64
    TOTAL_TILES: tl.constexpr = NUM_BATCHES * tiles_per_batch  # 65536

    sm_id = tl.program_id(0)
    tile_id = sm_id

    while tile_id < TOTAL_TILES:
        idx_q = tile_id // tiles_per_batch
        tile_mn = tile_id % tiles_per_batch
        pid_m = tile_mn // grid_n
        pid_n = tile_mn % grid_n

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        A = arg_A + (rm[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq)
        B = arg_B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn + idx_q * stride_bq)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(K, 0, -BLOCK_K):
            a = tl.load(A)
            b = tl.load(B)
            acc += tl.dot(a, b, allow_tf32=False)
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk

        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)
        xindex = idx_n + 512 * idx_m + 262144 * idx_q
        tl.store(out_ptr0 + xindex, acc, mask=mask)
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
