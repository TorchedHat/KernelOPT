import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_13(arg_A, arg_B, out_ptr1):
    BLOCK_M: tl.constexpr = 2
    BLOCK_N: tl.constexpr = 32
    BLOCK_K: tl.constexpr = 32
    M = 2
    N = 144
    K = 576
    pid_tile = tl.program_id(0)   # 0..4: which (m,n) output tile
    pid_k    = tl.program_id(1)   # 0..17: which K-split
    grid_n = tl.cdiv(N, BLOCK_N)  # =5
    pid_m = pid_tile // grid_n
    pid_n = pid_tile % grid_n
    k_start = pid_k * BLOCK_K
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_mask = offs_k[None, :] < (K - k_start)
    xindex_a = (offs_k[None, :] + k_start) + 576 * rm[:, None]
    a = tl.load(arg_A + xindex_a, mask=a_mask, other=0.0)
    b_mask = offs_k[:, None] < (K - k_start)
    xindex_b = rn[None, :] + 144 * (offs_k[:, None] + k_start)
    b = tl.load(arg_B + xindex_b, mask=b_mask, other=0.0)
    acc = tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    xindex_out = idx_n + 144 * idx_m
    tl.atomic_add(out_ptr1 + xindex_out, acc, mask=mask)


@triton.jit

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (5, 1, 1)
    triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_13[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 576, 1, 1), (576, 1, 576, 576), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((144, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 144, 1, 1), (144, 1, 144, 144), device='cuda:0', dtype=torch.float32)
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
