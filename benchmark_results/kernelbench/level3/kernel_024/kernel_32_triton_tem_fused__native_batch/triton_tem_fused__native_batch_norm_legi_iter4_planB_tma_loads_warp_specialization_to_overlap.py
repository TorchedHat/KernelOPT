import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=4,  num_stages=3),
        triton.Config({}, num_warps=4,  num_stages=4),
        triton.Config({}, num_warps=4,  num_stages=5),
        triton.Config({}, num_warps=8,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=3),
        triton.Config({}, num_warps=8,  num_stages=4),
        triton.Config({}, num_warps=8,  num_stages=5),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=4),
        triton.Config({}, num_warps=16, num_stages=5),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_32(arg_A, arg_B, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 128

    M = 2
    N = 1408
    K = 384

    if M * N == 0:
        return

    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # Build block pointers for A (shape M x K, row-major: strides (384, 1))
    a_block_ptr = tl.make_block_ptr(
        base=arg_A,
        shape=(M, K),
        strides=(384, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # Build block pointers for B (shape K x N, strides (1, 384) — transposed access)
    b_block_ptr = tl.make_block_ptr(
        base=arg_B,
        shape=(K, N),
        strides=(1, 384),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # Epilogue: batch norm + ReLU
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    tmp0  = tl.load(in_ptr2 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp2  = tl.load(in_ptr3 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
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
    tmp17 = (tmp16 / tmp8)
    tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 1408 * idx_m, [BLOCK_M, BLOCK_N])), tmp17, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    grid = (44, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_32[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((2, 384, 1, 1), (384, 1, 384, 384), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1408, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1408,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1408,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1408,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((1408,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((2, 1408, 1, 1), (1408, 1, 2816, 2816), device='cuda:0', dtype=torch.float32)
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
