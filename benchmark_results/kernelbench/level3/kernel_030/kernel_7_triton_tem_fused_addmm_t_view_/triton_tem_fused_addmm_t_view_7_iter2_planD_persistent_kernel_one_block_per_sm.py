import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_t_view_7(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 2
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 169
    N = 512
    K = 2
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 2
    stride_ak = 1
    stride_bk = 1
    stride_bn = 2

    sm_id = tl.program_id(0).to(INDEX_DTYPE)
    grid_n = tl.cdiv(N, BLOCK_N)   # 16
    num_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)  # 11 * 16 = 176
    num_sms = tl.num_programs(0)   # 132

    tile_id = sm_id
    while tile_id < num_tiles:
        pid_m = tile_id // grid_n
        pid_n = tile_id % grid_n

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        offs_a_m = rm % M
        offs_b_n = rn % N
        offs_k = tl.arange(0, BLOCK_K)

        # Load A: shape [BLOCK_M, BLOCK_K]
        # A[m, k] = A_ptr + k + 2*m  (stride_ak=1, stride_am=2)
        a_k_vals = offs_k[None, :]           # [1, BLOCK_K]
        a_m_vals = offs_a_m[:, None]         # [BLOCK_M, 1]
        xindex_a = a_k_vals + 2 * a_m_vals   # [BLOCK_M, BLOCK_K]
        a = tl.load(A + xindex_a)            # [BLOCK_M, BLOCK_K], no mask: EVEN_K=True

        # Load B: shape [BLOCK_K, BLOCK_N]
        # B[k, n] = B_ptr + n + 2*k  (stride_bk=1, stride_bn=2)
        b_k_vals = offs_k[:, None]           # [BLOCK_K, 1]
        b_n_vals = offs_b_n[None, :]         # [1, BLOCK_N]
        xindex_b = b_n_vals + 2 * b_k_vals   # [BLOCK_K, BLOCK_N]
        b = tl.load(B + tl.broadcast_to(xindex_b, [BLOCK_K, BLOCK_N]))  # [BLOCK_K, BLOCK_N]

        # Manual outer-product accumulation since BLOCK_K=2 < 16 (tl.dot requires K>=16)
        # acc[m, n] = sum_k a[m, k] * b[k, n]
        # Expand dims and broadcast: [BLOCK_M, BLOCK_K, 1] * [1, BLOCK_K, BLOCK_N] -> sum over axis=1
        a_3d = tl.expand_dims(a, 2)          # [BLOCK_M, BLOCK_K, 1]
        b_3d = tl.expand_dims(b, 0)          # [1, BLOCK_K, BLOCK_N]
        acc = tl.sum(a_3d * b_3d, axis=1)    # [BLOCK_M, BLOCK_N]
        acc = acc.to(ACC_TYPE)

        # Store with boundary mask
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)
        xindex = idx_n + 512 * idx_m
        tl.store(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask=mask)

        tile_id += num_sms

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (96, 1, 1)
    triton_tem_fused_addmm_t_view_7[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((1, 13, 13, 2), (338, 26, 2, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 2), (2, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((169, 512), (512, 1), device='cuda:0', dtype=torch.float32)
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
