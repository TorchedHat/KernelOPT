import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_max_pool2d_with_indices_mean_relu_t_14(in_ptr0, arg_A, arg_B, out_ptr0):
    M = 8
    N = 10
    K = 256
    BLOCK_K : tl.constexpr = 256  # power-of-2, covers all K in one shot

    pid = tl.program_id(0)
    # Each program handles exactly one output element C[m, n]
    m = pid // N   # row index in [0, M)
    n = pid % N    # col index in [0, N)

    # Load A[m, :] — shape [K], A has shape (8,256) strides (256,1)
    # A[m, k] = arg_A[m*256 + k]
    k_offs = tl.arange(0, BLOCK_K)          # [0..255]
    a = tl.load(arg_A + m * 256 + k_offs)   # A[m, :], contiguous in k

    # Load B[n, :] — shape [K], B has shape (10,256) strides (256,1)
    # B[n, k] = arg_B[n*256 + k]
    b = tl.load(arg_B + n * 256 + k_offs)   # B[n, :], contiguous in k

    # Scalar dot product: sum_k A[m,k] * B[n,k]
    # Cast to float32 for accumulation
    a_f32 = a.to(tl.float32)
    b_f32 = b.to(tl.float32)
    acc = tl.sum(a_f32 * b_f32, axis=0)    # scalar float32

    # Add bias: in_ptr0 has shape (10,) stride (1,)
    # bias[n] = in_ptr0[n]
    bias = tl.load(in_ptr0 + n)             # scalar float32
    result = acc + bias.to(tl.float32)

    # Store C[m,n]: out_ptr0 has shape (8,10) strides (10,1)
    # C[m,n] = out_ptr0[m*10 + n]
    tl.store(out_ptr0 + m * 10 + n, result)

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
