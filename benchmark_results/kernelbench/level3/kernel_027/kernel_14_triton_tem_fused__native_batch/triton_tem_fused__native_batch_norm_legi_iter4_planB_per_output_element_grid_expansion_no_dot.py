import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_max_pool2d_with_indices_mean_relu_t_14(
    in_ptr0, arg_A, arg_B, out_ptr0
):
    M = 8
    N = 10
    K = 256
    BLOCK_K : tl.constexpr = 256
    pid = tl.program_id(0)              # 0..79
    m = pid // N                        # row 0..7
    n = pid % N                         # col 0..9
    k_offs = tl.arange(0, BLOCK_K)     # [0..255]
    a_vals = tl.load(arg_A + m * K + k_offs)   # load A[m,:] contiguous row
    b_vals = tl.load(arg_B + n * K + k_offs)   # load B[n,:] contiguous row
    acc = tl.sum(a_vals * b_vals, axis=0)       # scalar - do NOT index with [0]
    bias = tl.load(in_ptr0 + n)                 # scalar bias[n]
    tl.store(out_ptr0 + m * N + n, acc + bias)  # scalar store

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
