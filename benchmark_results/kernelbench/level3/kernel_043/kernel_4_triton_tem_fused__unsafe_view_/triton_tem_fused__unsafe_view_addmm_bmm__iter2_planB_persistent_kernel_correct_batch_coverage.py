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
    ],
    key=[],
)
@triton.jit
def triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4(arg_A, arg_B, out_ptr0):
    # All problem constants hardcoded
    M: tl.constexpr = 512
    N: tl.constexpr = 512
    K: tl.constexpr = 96
    NUM_BATCHES: tl.constexpr = 1024
    NUM_SMS: tl.constexpr = 132
    BLOCK_M: tl.constexpr = 32
    BLOCK_N: tl.constexpr = 32
    BLOCK_K: tl.constexpr = 32

    # Persistent kernel: each SM loops over multiple tiles
    sm_id = tl.program_id(0)
    tile_id = sm_id

    grid_m = tl.cdiv(M, BLOCK_M)   # 512/32 = 16
    grid_n = tl.cdiv(N, BLOCK_N)   # 512/32 = 16
    tiles_per_batch = grid_m * grid_n  # 256
    total_tiles = tiles_per_batch * NUM_BATCHES  # 256 * 1024 = 262144

    while tile_id < total_tiles:
        # Decode flat tile_id into (batch, tile_m, tile_n)
        idx_q = tile_id // tiles_per_batch
        tile_mn = tile_id % tiles_per_batch
        pid_m = tile_mn // grid_n
        pid_n = tile_mn % grid_n

        # Compute row/col ranges for this tile
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        # Setup pointers
        # A: strides (49152, 96, 1) -> batch, row, col
        A_ptr = arg_A + idx_q * 49152 + rm[:, None] * 96 + rk[None, :] * 1
        # B: strides (49152, 512, 1) -> batch, row, col
        B_ptr = arg_B + idx_q * 49152 + rk[:, None] * 512 + rn[None, :] * 1

        # Accumulate over K dimension (K=96, BLOCK_K=32 -> 3 iterations)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(K, 0, -BLOCK_K):
            a = tl.load(A_ptr)
            b = tl.load(B_ptr)
            acc += tl.dot(a, b, allow_tf32=False)
            A_ptr += BLOCK_K * 1
            B_ptr += BLOCK_K * 512

        # Store result with boundary mask
        rm_store = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn_store = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rm_store[:, None] < M) & (rn_store[None, :] < N)
        xindex = rn_store[None, :] + 512 * rm_store[:, None] + 262144 * idx_q
        tl.store(out_ptr0 + xindex, acc, mask=mask)

        # Stride to next tile for this SM
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
