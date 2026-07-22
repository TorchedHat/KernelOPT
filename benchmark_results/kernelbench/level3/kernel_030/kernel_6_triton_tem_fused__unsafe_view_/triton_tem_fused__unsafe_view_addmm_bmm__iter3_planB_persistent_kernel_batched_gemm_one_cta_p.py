import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__unsafe_view_addmm_bmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_6(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32

    M = 49
    N = 49
    K = 32

    stride_aq = 1568
    stride_am = 32
    stride_ak = 1

    stride_bq = 1568
    stride_bk = 49
    stride_bn = 1

    NUM_SMS = 132
    TOTAL_BATCHES = 1920
    NUM_HEADS = 3

    # Persistent kernel: one CTA per SM, each processes multiple batch slices
    cta_id = tl.program_id(0).to(INDEX_DTYPE)
    slices_per_cta = (TOTAL_BATCHES + NUM_SMS - 1) // NUM_SMS  # = 15

    batch_start = cta_id * slices_per_cta
    batch_end = tl.minimum(batch_start + slices_per_cta, TOTAL_BATCHES)

    # Since M=49 < BLOCK_M=64 and N=49 < BLOCK_N=64, pid_m=0 and pid_n=0 always
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # ram and rbn: since stride_am=32=K and stride_ak=1, use max_contiguous path
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    # stride_bk=49=N and stride_bn=1, so use max_contiguous path
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    for flat_batch in tl.range(batch_start, batch_end):
        pid_head = flat_batch % NUM_HEADS    # head index: 0..2
        pid_outer = flat_batch // NUM_HEADS  # outer batch index: 0..639

        A_ptr = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + pid_head * stride_aq + pid_outer * 4704)
        B_ptr = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + pid_head * stride_bq + pid_outer * 4704)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for k in range(K, 0, -BLOCK_K):
            if EVEN_K:
                a = tl.load(A_ptr)
                b = tl.load(B_ptr)
            else:
                a = tl.load(A_ptr, mask=rk[None, :] < k, other=0.)
                b = tl.load(B_ptr, mask=rk[:, None] < k, other=0.)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_ptr += BLOCK_K * stride_ak
            B_ptr += BLOCK_K * stride_bk

        xindex = idx_n + 49 * idx_m + 2401 * flat_batch
        tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (1, 1, 1)
    triton_tem_fused__unsafe_view_addmm_bmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_transpose_view_zeros_like_6[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((640, 3, 49, 32), (4704, 1568, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((640, 3, 32, 49), (4704, 1568, 49, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1920, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
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
