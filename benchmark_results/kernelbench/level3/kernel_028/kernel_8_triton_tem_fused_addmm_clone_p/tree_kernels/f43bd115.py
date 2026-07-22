import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_clone_permute_t_view_8(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 394
    N = 512
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

    grid_m = (M + BLOCK_M - 1) // BLOCK_M  # 13
    grid_n = (N + BLOCK_N - 1) // BLOCK_N  # 8
    num_tiles = grid_m * grid_n             # 104

    # Persistent kernel: each SM loops over multiple tiles
    num_programs = tl.num_programs(0).to(INDEX_DTYPE)
    sm_id = tl.program_id(0).to(INDEX_DTYPE)
    tile_id = sm_id  # each SM starts at its own tile index

    while tile_id < num_tiles:
        # Decode flat tile_id -> (pid_m, pid_n) with L2-friendly GROUP_M swizzle
        width = GROUP_M * grid_n
        group_id = tile_id // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (tile_id % group_size)
        pid_n = (tile_id % width) // group_size
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_a_m = rm % M
        offs_b_n = rn % N
        offs_k = tl.arange(0, BLOCK_K)

        # acc MUST be inside the while-loop (re-initialized per tile)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
            xindex = a_k_idx_vals + 512 * offs_a_m[:, None]
            a = tl.load(A + xindex)

            b_k_offs = offs_k[:, None] + (k_idx * BLOCK_K)
            b = tl.load(B + b_k_offs + 512 * offs_b_n[None, :])

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

        # Store this tile's result
        rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm2[:, None]
        idx_n = rn2[None, :]
        mask = (idx_m < M) & (idx_n < N)
        xindex = idx_n + 512 * idx_m
        tl.store(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask)

        tile_id += num_programs  # stride to next tile for this SM

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (112, 1, 1)
    triton_tem_fused_addmm_clone_permute_t_view_8[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 197, 8, 64), (100864, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((394, 512), (512, 1), device='cuda:0', dtype=torch.float32)
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
