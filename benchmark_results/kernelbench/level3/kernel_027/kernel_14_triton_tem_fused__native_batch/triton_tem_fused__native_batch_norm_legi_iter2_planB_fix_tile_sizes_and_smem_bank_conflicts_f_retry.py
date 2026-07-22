import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_max_pool2d_with_indices_mean_relu_t_14(in_ptr0, arg_A, arg_B, out_ptr0):
    # M=8, N=10, K=256 — one CTA per output element
    M : tl.constexpr = 8
    N : tl.constexpr = 10
    K : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 32

    # Each program handles one (m, n) output element
    pid = tl.program_id(0)   # 0..M*N-1 = 0..79
    m = pid // N              # row index in [0,7]
    n = pid % N               # col index in [0,9]

    # Accumulate A[m,:] dot B[:,n] over K=256
    # A: shape [8,256], strides (256,1) -> A[m,k] = arg_A + m*256 + k
    # B: shape [10,256], strides (256,1) -> B[n,k] = arg_B + n*256 + k
    # (B is stored as [N,K] with stride_bn=256, stride_bk=1)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for k_start in tl.static_range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        a_vals = tl.load(arg_A + m * 256 + k_offs)  # A[m, k_start:k_start+BLOCK_K]
        b_vals = tl.load(arg_B + n * 256 + k_offs)  # B[n, k_start:k_start+BLOCK_K]
        acc += a_vals * b_vals

    result = tl.sum(acc, axis=0)

    # Add bias[n] and store to out[m,n]
    bias = tl.load(in_ptr0 + n)
    tl.store(out_ptr0 + m * N + n, result + bias)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (1, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_max_pool2d_with_indices_mean_relu_t_14[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((10,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8, 10), (10, 1), device='cuda:0', dtype=torch.float32)
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
