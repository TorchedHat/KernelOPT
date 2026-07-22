import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({}, num_warps=1, num_stages=1),
    triton.Config({}, num_warps=2, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=1, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=2),
], key=[])
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_22(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    ALLOW_TF32 : tl.constexpr = True
    A = arg_A
    B = arg_B

    M = 2
    N = 1152
    K = 192
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    pid = tl.program_id(0).to(tl.int32)
    num_progs = tl.num_programs(0).to(tl.int32)
    grid_m = tl.cdiv(M, BLOCK_M)   # = 1
    grid_n = tl.cdiv(N, BLOCK_N)   # = 36
    num_tiles = grid_m * grid_n     # = 36

    tile_id = pid  # start at this block's SM index
    while tile_id < num_tiles:
        pid_m = tile_id // grid_n   # = 0 always (M=2, BLOCK_M=2)
        pid_n = tile_id % grid_n    # 0..35

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
            a = tl.load(A + (idx_n_a + 192 * idx_m))

            idx_m_b = b_k_idx_vals
            idx_n_b = offs_b_n[None, :]
            b = tl.load(B + tl.broadcast_to(idx_m_b + 192 * idx_n_b, [BLOCK_K, BLOCK_N]))

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

        # Epilogue: BN + ReLU
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)

        tmp0  = tl.load(in_ptr2 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp2  = tl.load(in_ptr3 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr4 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]), mask, eviction_policy='evict_last')
        tmp1  = acc - tmp0
        tmp3  = tl.full([1], 1e-05, tl.float32)
        tmp4  = tmp2 + tmp3
        tmp5  = tl.sqrt_rn(tmp4)
        tmp6  = tl.full([1], 1, tl.int32)
        tmp7  = (tmp6 / tmp5)
        tmp8  = tl.full([1], 1.0, tl.float32)
        tmp9  = tmp7 * tmp8
        tmp10 = tmp1 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tl.full([1], 0, tl.int32)
        tmp16 = tl.maximum(tmp15, tmp14)
        tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 1152 * idx_m, [BLOCK_M, BLOCK_N])), tmp16, mask)

        tile_id += num_progs  # stride by number of SMs (132)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    grid = (36, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_22[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((2, 192, 1, 1), (192, 1, 192, 192), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((1152,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((2, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
