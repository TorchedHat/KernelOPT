import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_addmm_gelu_t_view_56(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 490
    N = 768
    K = 3072
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3072
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3072

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 3072*idx_m
        a = tl.load(A + (xindex))

        # B has shape (768, 3072) with strides (3072, 1): B[n, k] = B + n*3072 + k
        # Load B tile: shape [BLOCK_K, BLOCK_N]
        # b_k: K-dimension offsets, shape [BLOCK_K, 1]
        # b_n: N-dimension offsets, shape [1, BLOCK_N]
        b_k = offs_k[:, None] + (k_idx * BLOCK_K)   # shape [BLOCK_K, 1]
        b_n = offs_b_n[None, :]                       # shape [1, BLOCK_N]
        # Physical address: B[n, k] = B + n*3072 + k*1
        # = B + b_n * 3072 + b_k
        # Consecutive threads vary along N (b_n), accessing b_n*3072 + b_k
        # With b_n contiguous, stride between consecutive threads = 3072 elements
        # To make consecutive threads access contiguous memory, we need to
        # iterate over N as the fast dimension: B[n, k] with n varying fast
        # Since B is stored row-major [N, K], B[n, k] = B + n*3072 + k
        # For coalesced access, we want consecutive threads to access consecutive n values
        # That means: thread i accesses B[n_base + i, k] = B + (n_base+i)*3072 + k
        # This is stride 3072 between threads — still strided in global memory
        # BUT the bank conflict is in SHARED memory, not global memory
        # The fix: load B transposed so that the shared memory layout is coalesced
        # Load B[n, k] = B + n*3072 + k where n varies along columns (BLOCK_N dim)
        # and k varies along rows (BLOCK_K dim)
        # For tl.dot(a, b): a=[BLOCK_M, BLOCK_K], b=[BLOCK_K, BLOCK_N]
        # b[k_local, n_local] = B[b_n[n_local], b_k[k_local]] = B + b_n*3072 + b_k
        b = tl.load(B + (b_n * 3072 + b_k))

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 768*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (96, 1, 1)
    triton_tem_fused_addmm_gelu_t_view_56[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 49, 3072), (150528, 3072, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 768), (768, 1), device='cuda:0', dtype=torch.float32)
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
