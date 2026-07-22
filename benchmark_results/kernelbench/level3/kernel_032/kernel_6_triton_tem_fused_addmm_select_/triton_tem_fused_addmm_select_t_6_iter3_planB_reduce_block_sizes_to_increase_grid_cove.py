import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_select_t_6(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 8
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 10
    N = 1000
    K = 128
    if M * N == 0:
        return

    grid_m = (M + BLOCK_M - 1) // BLOCK_M   # = 2
    grid_n = (N + BLOCK_N - 1) // BLOCK_N   # = 63
    total_tiles = grid_m * grid_n            # = 126
    num_sms = tl.num_programs(0)
    sm_id = tl.program_id(0).to(INDEX_DTYPE)

    for tile_id in range(sm_id, total_tiles, num_sms):
        pid = tile_id.to(INDEX_DTYPE)
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_a_m = rm % M
        offs_b_n = rn % N
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

            idx_m = offs_a_m[:, None]
            idx_n_a = a_k_idx_vals
            a = tl.load(A + tl.broadcast_to(idx_n_a + 256 * idx_m, [BLOCK_M, BLOCK_K]))

            idx_m_b = b_k_idx_vals
            idx_n_b = offs_b_n[None, :]
            b = tl.load(B + tl.broadcast_to(idx_m_b + 128 * idx_n_b, [BLOCK_K, BLOCK_N]))

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

        # rematerialize rm and rn to save registers
        rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m2 = rm2[:, None]
        idx_n2 = rn2[None, :]
        mask = (idx_m2 < M) & (idx_n2 < N)

        xindex = idx_n2 + 1000 * idx_m2
        tmp0 = tl.load(in_ptr0 + tl.broadcast_to(idx_n2, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tl.store(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc + tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (32, 1, 1)
    triton_tem_fused_addmm_select_t_6[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((1000,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 2, 128), (256, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
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
