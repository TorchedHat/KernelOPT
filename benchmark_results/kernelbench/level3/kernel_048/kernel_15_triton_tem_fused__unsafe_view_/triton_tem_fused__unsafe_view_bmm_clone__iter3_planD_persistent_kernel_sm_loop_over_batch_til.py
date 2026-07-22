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
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    TOTAL_BATCHES : tl.constexpr = 32768
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

    # Since BLOCK_M=M=64 and BLOCK_N=N=64, pid_m=0 and pid_n=0 always
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # ram: stride_am=16, stride_ak=1 => neither contiguous condition met => ram = rm % M
    ram = rm % M
    # rbn: stride_bk=64=N, stride_bn=1 => (stride_bk==N and stride_bn==1) => contiguous
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    # Persistent SM loop: each SM processes batch indices sm_id, sm_id+NUM_SMS, ...
    sm_id = tl.program_id(0).to(INDEX_DTYPE)
    idx_q = sm_id

    for _step in tl.range(0, TOTAL_BATCHES, NUM_SMS):
        if idx_q < TOTAL_BATCHES:
            # Reset accumulator at the TOP of each batch iteration
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

            # Set up pointers for this batch element
            A = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq)
            B = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq)

            # K-reduction loop
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

            # Store result
            idx_m = rm[:, None]
            idx_n = rn[None, :]
            mask = (idx_m < M) & (idx_n < N)
            xindex = idx_n + 64 * idx_m + 4096 * idx_q
            tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

        idx_q += NUM_SMS

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
