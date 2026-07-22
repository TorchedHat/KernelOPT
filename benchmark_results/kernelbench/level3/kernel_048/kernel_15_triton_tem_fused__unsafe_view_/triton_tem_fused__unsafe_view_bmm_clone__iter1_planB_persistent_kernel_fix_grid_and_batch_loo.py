import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    TOTAL_BATCH : tl.constexpr = 32768
    NUM_SMS : tl.constexpr = 132

    M = 64
    N = 64
    K = 16

    stride_aq = 1024
    stride_am = 16
    stride_ak = 1

    stride_bq = 1024
    stride_bk = 64
    stride_bn = 1

    # Persistent kernel: each SM processes a contiguous slice of batch indices
    sm_id = tl.program_id(0).to(INDEX_DTYPE)
    tiles_per_sm = (TOTAL_BATCH + NUM_SMS - 1) // NUM_SMS
    start_q = sm_id * tiles_per_sm
    end_q = tl.minimum(start_q + tiles_per_sm, TOTAL_BATCH)

    # Since M=N=64=BLOCK_M=BLOCK_N, pid_m=0 and pid_n=0 always
    pid_m = 0
    pid_n = 0
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # stride_am=16=K, stride_ak=1 => use max_contiguous hint
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    # stride_bk=64=N, stride_bn=1 => use max_contiguous hint
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    for idx_q in range(start_q, end_q):
        # Recompute A and B pointers for this batch element
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

        # Store result for this batch element
        xindex = idx_n + 64 * idx_m + 4096 * idx_q
        tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (1, 1, 1)
    triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2048, 2, 8, 64, 16, 1), (16384, 8192, 1024, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 2, 8, 16, 64, 1), (16384, 8192, 1024, 64, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32768, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
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
