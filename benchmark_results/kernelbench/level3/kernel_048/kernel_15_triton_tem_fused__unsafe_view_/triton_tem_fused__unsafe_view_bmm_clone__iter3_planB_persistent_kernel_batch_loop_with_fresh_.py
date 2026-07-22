import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__unsafe_view_bmm_clone_permute_slice_unsqueeze_view_15(arg_A, arg_B, out_ptr0):
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 16
    ALLOW_TF32: tl.constexpr = False
    ACC_TYPE: tl.constexpr = tl.float32
    TOTAL_BATCHES: tl.constexpr = 32768
    NUM_SMS: tl.constexpr = 132

    M = 64
    N = 64
    K = 16

    stride_aq = 1024
    stride_am = 16
    stride_ak = 1

    stride_bq = 1024
    stride_bk = 64
    stride_bn = 1

    sm_id = tl.program_id(0).to(tl.int32)
    tiles_per_sm = (TOTAL_BATCHES + NUM_SMS - 1) // NUM_SMS
    batch_start = sm_id * tiles_per_sm
    batch_end = tl.minimum(batch_start + tiles_per_sm, TOTAL_BATCHES)

    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    ram = rm % M
    rbn = rn % N

    for idx_q in range(batch_start, batch_end):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)  # CRITICAL: inside loop, reset per batch
        A_base = arg_A + ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq
        B_base = arg_B + rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq
        A_ptr = A_base
        B_ptr = B_base
        for k in range(K, 0, -BLOCK_K):
            a = tl.load(A_ptr)
            b = tl.load(B_ptr)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_ptr += BLOCK_K * stride_ak
            B_ptr += BLOCK_K * stride_bk
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)
        xindex = idx_n + 64 * idx_m + 4096 * idx_q
        tl.store(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask)

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
