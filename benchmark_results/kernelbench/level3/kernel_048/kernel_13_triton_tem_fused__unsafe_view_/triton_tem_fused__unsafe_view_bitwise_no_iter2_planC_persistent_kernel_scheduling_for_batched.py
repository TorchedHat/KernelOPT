import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13(arg_A, arg_B, out_ptr0):
    BLOCK_M: tl.constexpr = 16
    BLOCK_N: tl.constexpr = 32
    BLOCK_K: tl.constexpr = 16
    ACC_TYPE: tl.constexpr = tl.float32
    ALLOW_TF32: tl.constexpr = False

    M = 3
    N = 1024
    K = 3

    stride_aq = 9
    stride_am = 3
    stride_ak = 1

    stride_bq = 3072
    stride_bk = 1024
    stride_bn = 1

    # grid_m = ceil(3/16) = 1, grid_n = ceil(1024/32) = 32
    grid_m = 1
    grid_n = 32
    tiles_per_batch = grid_m * grid_n   # 32
    N_BATCH = 16384
    total_tiles = N_BATCH * tiles_per_batch  # 524288
    num_programs = tl.num_programs(0)        # actual grid size (528)

    cta_id = tl.program_id(0)
    tile_idx = cta_id

    while tile_idx < total_tiles:
        idx_q = tile_idx // tiles_per_batch
        local = tile_idx % tiles_per_batch
        pid_m = local // grid_n
        pid_n = local % grid_n

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        ram = rm % M
        rbn = rn % N

        A = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq)
        B = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for k in range(K, 0, -BLOCK_K):
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk

        rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm2[:, None]
        idx_n = rn2[None, :]
        mask = (idx_m < M) & (idx_n < N)
        xindex = idx_n + 1024 * idx_m + 3072 * idx_q
        tl.store(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask)

        tile_idx += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (32, 1, 1)
    triton_tem_fused__unsafe_view_bitwise_not_bmm_cat_clone_exp_masked_fill_ones_permute_sub_tril_unsqueeze_view_zeros_like_13[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2048, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8, 3, 64, 16, 1), (24576, 3072, 1024, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16384, 3, 1024), (3072, 1024, 1), device='cuda:0', dtype=torch.float32)
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
