import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=2, num_stages=2),
], key=[], reset_to_zero=['out_ptr0'])
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_20(arg_A, arg_B, out_ptr0):
    # Split-K constants
    SPLIT_K : tl.constexpr = 22
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32

    M = 2
    N = 192
    K = 864

    if M * N == 0:
        return

    pid_mn = tl.program_id(0)
    pid_k  = tl.program_id(1)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n

    K_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * K_per_split
    k_end   = tl.minimum(k_start + K_per_split, K)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_a_m = rm % M
    offs_b_n = rn % N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(K_per_split, BLOCK_K)):
        k_off = k_start + k_idx * BLOCK_K
        offs_k = tl.arange(0, BLOCK_K)
        k_mask = (k_off + offs_k) < k_end

        # A[m, k] = arg_A + m * 864 + k  (stride_am=864, stride_ak=1)
        a = tl.load(arg_A + offs_a_m[:, None] * 864 + (k_off + offs_k)[None, :],
                    mask=k_mask[None, :], other=0.0)

        # B[k, n] = arg_B + n * 864 + k  (stride_bk=1, stride_bn=864)
        b = tl.load(arg_B + offs_b_n[None, :] * 864 + (k_off + offs_k)[:, None],
                    mask=k_mask[:, None], other=0.0)

        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    out_ptrs = out_ptr0 + idx_m * 192 + idx_n

    tl.atomic_add(out_ptrs, acc, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (6, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_20[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 864, 1, 1), (864, 1, 864, 864), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 192), (192, 1), device='cuda:0', dtype=torch.float32)
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
