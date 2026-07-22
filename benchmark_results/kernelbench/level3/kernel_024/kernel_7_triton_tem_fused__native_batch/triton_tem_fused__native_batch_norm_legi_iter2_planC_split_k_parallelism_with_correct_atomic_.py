import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_7(arg_A, arg_B, out_ptr1):
    BLOCK_M : tl.constexpr = 2
    BLOCK_N : tl.constexpr = 8
    BLOCK_K : tl.constexpr = 32
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32

    M = 2
    N = 24
    K = 96

    grid_m = (M + BLOCK_M - 1) // BLOCK_M  # = 1
    grid_n = (N + BLOCK_N - 1) // BLOCK_N  # = 3

    pid = tl.program_id(0)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # Row/col indices for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # K loop over all K=96 elements in chunks of BLOCK_K=32
    for k_idx in range(0, 3):
        k_off = k_idx * BLOCK_K + offs_k  # absolute k indices

        # A-matrix: A[m, k] at offset k + 96*m
        idx_m_a = rm[:, None]
        idx_k_a = k_off[None, :]
        a_mask = (idx_m_a < M) & (idx_k_a < K)
        a = tl.load(arg_A + (idx_k_a + 96 * idx_m_a), mask=a_mask, other=0.0)

        # B-matrix: arg_1 has shape (24, 96, 1, 1) with strides (96, 1, 1, 1)
        # B[n, k] is at offset n*96 + k
        idx_k_b = k_off[:, None]
        idx_n_b = rn[None, :]
        b_mask = (idx_k_b < K) & (idx_n_b < N)
        b = tl.load(arg_B + (idx_k_b + 96 * idx_n_b), mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # Write output with ReLU
    idx_m_out = rm[:, None]
    idx_n_out = rn[None, :]
    out_mask = (idx_m_out < M) & (idx_n_out < N)
    out_idx = idx_n_out + 24 * idx_m_out
    tmp0 = tl.full([1], 0, tl.int32)
    result = tl.maximum(tmp0, acc)
    tl.store(out_ptr1 + out_idx, result, mask=out_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (1, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_7[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 96, 1, 1), (96, 1, 96, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 24, 1, 1), (24, 1, 24, 24), device='cuda:0', dtype=torch.float32)
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
