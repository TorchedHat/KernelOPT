import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_convolution_1(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 8
    BLOCK_K : tl.constexpr = 4
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 8388608
    N = 6
    K = 3
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3

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

        # Load A as [BLOCK_K, BLOCK_M]: M is fast dim (stride_am=3 but M index is innermost
        # for warp lanes), making adjacent threads access addresses 1 apart (stride_ak=1 is
        # the K stride, but here we load along M with stride_am=3... wait, we want M contiguous).
        # A[m, k] = A + m * stride_am + k * stride_ak = A + m*3 + k*1
        # Load a_km[k_local, m] = A[m, k_global]:
        #   address = A + offs_a_m[None,:] * stride_am + k_cur[:,None] * stride_ak
        #           = A + offs_a_m[None,:] * 3 + k_cur[:,None] * 1
        # Shape [BLOCK_K, BLOCK_M]: adjacent threads (varying m) access addresses 3 apart.
        # BUT: the M dimension has stride_am=3 in the original tensor, so loading along M
        # still has stride 3. The coalescing benefit comes from the fact that within a warp,
        # threads vary along the M dimension (BLOCK_M=128 >> BLOCK_K=4), so the innermost
        # tl.arange maps to M. Adjacent threads access m, m+1, m+2... at stride 3 each.
        # This is the same as before for M. The key insight from the plan: stride_ak=1 means
        # K elements are contiguous. Loading [BLOCK_K, BLOCK_M] with M as the fast (column)
        # dimension means the warp's 32 threads cover 32 consecutive M values, each separated
        # by stride_am=3 floats. That's still strided. However, the plan says M is stride-1
        # in the flattened view... Let me re-read: offs_a_m[None,:]*1 means stride_am=1?
        # No - the plan hint has a bug. The correct address is:
        # A + offs_a_m[None,:] * stride_am + k_cur[:,None] * stride_ak
        # = A + offs_a_m[None,:] * 3 + k_cur[:,None] * 1
        # This loads [BLOCK_K, BLOCK_M] with M varying fastest (innermost dim of warp).
        # Adjacent warp threads access m, m+1 -> addresses differ by stride_am=3 floats.
        # The plan claims this is coalesced because stride_ak=1 and we load along K first...
        # Actually the coalescing improvement: in the original load [BLOCK_M, BLOCK_K],
        # the warp's 32 threads cover the BLOCK_M dimension (rows), each row separated by
        # stride_am=3. In the transposed load [BLOCK_K, BLOCK_M], the warp's 32 threads
        # still cover M values with stride 3. The difference is that now BLOCK_K=4 rows
        # are loaded with K as the outer dim (stride_ak=1 between K elements), so consecutive
        # K loads are at stride 1 - but that's the outer loop, not the warp dimension.
        # The real benefit: in [BLOCK_K, BLOCK_M] layout, the 128 M-elements are the fast
        # dimension. A warp of 32 threads covers 32 consecutive M indices. The addresses are
        # offs_a_m[0..31] * 3 + k_cur. Since offs_a_m = pid_m*128 + [0..127], consecutive
        # M values give addresses that differ by 3 floats = 12 bytes. This is still strided.
        # BUT: the original [BLOCK_M, BLOCK_K] load has BLOCK_K=4 as the fast dim (innermost
        # tl.arange). A warp of 32 threads covers K indices [0,1,2,3] repeated 8 times across
        # M. Adjacent threads in K differ by stride_ak=1 (contiguous!) but then jump by
        # stride_am=3 for the next M row. So actually the original load IS somewhat coalesced
        # in K. The transposed load makes M the fast dim which is stride 3 - worse?
        # I need to just implement what the plan says correctly and trust the index math.
        # The plan's formula: A + offs_a_m[None,:] * 1 + k_cur * stride_am
        # This would be: A + offs_a_m[None,:] * 1 + k_cur * 3
        # That treats offs_a_m as stride-1 and k as stride-3. But A[m,k] = A + m*3 + k*1.
        # So offs_a_m[None,:]*1 + k_cur*3 = m + 3k which is WRONG (swapped strides).
        # The correct formula must be: offs_a_m[None,:]*3 + k_cur*1 = 3m + k = A[m,k]. CORRECT.
        k_cur = offs_k[:, None] + k_idx * BLOCK_K                   # [BLOCK_K, 1]
        a_k_mask = k_cur < K                                         # [BLOCK_K, 1]
        a_m_mask = offs_a_m[None, :] < M                            # [1, BLOCK_M]
        a_mask_km = a_k_mask & a_m_mask                             # [BLOCK_K, BLOCK_M]
        # Correct address: A[m, k] = A + m * stride_am + k * stride_ak = A + m*3 + k*1
        # Load as [BLOCK_K, BLOCK_M]: offs_a_m varies along dim-1 (fast/warp dim)
        a_km = tl.load(A + offs_a_m[None, :] * stride_am + k_cur * stride_ak,
                       mask=a_mask_km, other=0.0)                   # [BLOCK_K, BLOCK_M]
        a = tl.trans(a_km)                                          # [BLOCK_M, BLOCK_K]

        # Load B as [BLOCK_N, BLOCK_K] so K (stride=1) is the fast/inner dim
        b_k_cur = offs_k[None, :] + k_idx * BLOCK_K   # shape [1, BLOCK_K]
        b_n_mask = offs_b_n[:, None] < N               # shape [BLOCK_N, 1]
        b_k_mask = b_k_cur < K                         # shape [1, BLOCK_K]
        b_mask_nk = b_n_mask & b_k_mask
        # B[n, k]: stride_bn=3 for n, stride_bk=1 for k
        b_nk = tl.load(B + offs_b_n[:, None] * 3 + b_k_cur * 1,
                       mask=b_mask_nk, other=0.0)      # shape [BLOCK_N, BLOCK_K]
        b = tl.trans(b_nk)                             # shape [BLOCK_K, BLOCK_N] for matmul

        # Explicit FMA: acc[m, n] += sum_k a[m, k] * b[k, n]
        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N]
        # Use broadcasting: expand a to [BLOCK_M, BLOCK_K, 1], b to [1, BLOCK_K, BLOCK_N]
        # then sum over K axis
        acc += tl.sum(
            tl.reshape(a, (BLOCK_M, BLOCK_K, 1)) * tl.reshape(b, (1, BLOCK_K, BLOCK_N)),
            axis=1
        )

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 6*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (65536, 1, 1)
    triton_tem_fused_convolution_1[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((6,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 3, 256, 256), (196608, 1, 768, 3), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((6, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8388608, 6), (6, 1), device='cuda:0', dtype=torch.float32)
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
