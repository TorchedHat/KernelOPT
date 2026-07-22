import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_max_pool2d_with_indices_mean_relu_t_14(in_ptr0, arg_A, arg_B, out_ptr0):
    M: tl.constexpr = 8
    N: tl.constexpr = 10
    K: tl.constexpr = 256
    BLOCK_K: tl.constexpr = 256

    pid_m = tl.program_id(0)  # one CTA per output row, pid_m in [0, M)
    k_offs = tl.arange(0, BLOCK_K)  # [256]
    # Load full row A[pid_m, :] once into registers
    a_row = tl.load(arg_A + pid_m * K + k_offs)  # [256]
    # Iterate over all N output columns at compile time
    for n in tl.static_range(N):
        # B[n, k] = arg_B + n*256 + k  (stride_bn=256, stride_bk=1)
        b_row = tl.load(arg_B + n * K + k_offs)  # [256]
        dot_val = tl.sum(a_row * b_row, axis=0)   # scalar
        bias = tl.load(in_ptr0 + n)               # scalar
        tl.store(out_ptr0 + pid_m * N + n, dot_val + bias)

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
