import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_max_pool2d_with_indices_mean_relu_t_14(in_ptr0, arg_A, arg_B, out_ptr0):
    # Each program handles one (m, n) output element
    # Grid: (M*N,) = (80,) — one CTA per output element
    pid = tl.program_id(0)
    M = 8
    N = 10
    K = 256
    BLOCK_K: tl.constexpr = 256

    m = pid // N
    n = pid % N

    # Load row m of A: shape [K]
    k_offs = tl.arange(0, BLOCK_K)
    a_ptrs = arg_A + m * 256 + k_offs
    a_mask = k_offs < K
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    # Load column n of B (B is [N=10, K=256] row-major, B[n, k] = B_ptr + n*256 + k)
    # We need B[:, n] which is B[n, k] for all k — but B is stored as B[n, k]
    # B^T column n = B row n: B_ptr + n*256 + k_offs
    b_ptrs = arg_B + n * 256 + k_offs
    b_mask = k_offs < K
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Dot product: sum over K
    acc = tl.sum(a * b, axis=0)

    # Load bias
    bias = tl.load(in_ptr0 + n)

    # Store result
    result = acc + bias
    tl.store(out_ptr0 + m * N + n, result)

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
