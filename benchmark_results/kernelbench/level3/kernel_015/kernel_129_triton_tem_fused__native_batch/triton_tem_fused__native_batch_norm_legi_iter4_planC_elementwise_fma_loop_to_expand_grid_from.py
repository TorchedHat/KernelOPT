import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_addmm_mean_relu_t_view_129(in_ptr0, arg_A, arg_B, out_ptr0):
    BLOCK_K: tl.constexpr = 32
    M = 10
    N = 10
    K = 1024

    pid = tl.program_id(0)
    m = pid // N   # row index in [0, 10)
    n = pid % N    # col index in [0, 10)

    # Accumulate C[m,n] = sum_k A[m,k] * B[k,n]
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k_base in range(0, K, BLOCK_K):
        k_offs = k_base + tl.arange(0, BLOCK_K)
        # A: shape (10,1024,1,1), strides (1024,1,...) -> A[m,k] = arg_A + m*1024 + k
        a = tl.load(arg_A + m * 1024 + k_offs)
        # B: shape (10,1024), strides (1024,1) -> B[n,k] = arg_B[n*1024 + k]
        b = tl.load(arg_B + n * 1024 + k_offs)
        acc += a * b

    result = tl.sum(acc, axis=0)  # scalar
    bias = tl.load(in_ptr0 + n)   # bias shape (10,) stride (1,)
    result = result + bias
    tl.store(out_ptr0 + m * 10 + n, result)  # out shape (10,10) stride (10,1)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (1, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_addmm_mean_relu_t_view_129[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((10,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 1024, 1, 1), (1024, 1, 10240, 10240), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
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
