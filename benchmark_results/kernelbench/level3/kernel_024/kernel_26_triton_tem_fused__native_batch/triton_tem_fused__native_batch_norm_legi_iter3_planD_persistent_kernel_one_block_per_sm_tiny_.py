import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=1, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=1, num_stages=4),
        triton.Config({}, num_warps=2, num_stages=4),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_26(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 32
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 288
    K = 1152
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1152
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1152

    # Persistent kernel: each SM loops over its assigned tiles
    sm_id = tl.program_id(0).to(INDEX_DTYPE)
    num_sms = tl.num_programs(0).to(INDEX_DTYPE)
    grid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = grid_n  # grid_m = 1 since M=2, BLOCK_M=2

    tile_id = sm_id
    while tile_id < total_tiles:
        pid_m = tile_id // grid_n
        pid_n = tile_id % grid_n
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
            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

            idx_m = offs_a_m[:, None]
            idx_n = a_k_idx_vals
            xindex = idx_n + 1152*idx_m
            a = tl.load(A + (xindex))

            idx_m = b_k_idx_vals
            idx_n = offs_b_n[None, :]
            xindex = idx_n + 288*idx_m
            b = tl.load(B + ((tl.broadcast_to(idx_m + 1152*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)

        # inductor generates a suffix
        xindex = idx_n + 288*idx_m
        tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr3 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
        tmp1 = acc - tmp0
        tmp3 = tl.full([1], 1e-05, tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tl.sqrt_rn(tmp4)
        tmp6 = tl.full([1], 1, tl.int32)
        tmp7 = (tmp6 / tmp5)
        tmp8 = tl.full([1], 1.0, tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp1 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 288*idx_m, [BLOCK_M, BLOCK_N])), tmp14, mask)

        tile_id += num_sms

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    grid = (9, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_26[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((2, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((288, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((288,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((288,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((288,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((288,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((2, 288, 1, 1), (288, 1, 288, 288), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6]

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
