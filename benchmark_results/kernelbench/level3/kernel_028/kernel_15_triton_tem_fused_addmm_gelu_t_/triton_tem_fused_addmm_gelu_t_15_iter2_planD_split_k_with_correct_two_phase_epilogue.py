import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def _splitk_matmul_kernel(A, B, out_ptr0, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_wm, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    grid_mn = grid_m * grid_n
    pid_sk = pid // grid_mn
    pid_mn = pid % grid_mn
    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_sk * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_off in range(0, tl.cdiv(k_per_split, BLOCK_K)):
        k_base = k_start + k_off * BLOCK_K
        k_mask = (rk + k_base) < k_end
        a_ptrs = A + rm[:, None] * stride_am + (rk[None, :] + k_base) * stride_ak
        b_ptrs = B + (rk[:, None] + k_base) * stride_bk + rn[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    tl.atomic_add(out_ptr0 + rm[:, None] * stride_wm + rn[None, :] * stride_wn, acc, mask=mask_out)


@triton.jit

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    grid = (1, 1, 1)
    _splitk_matmul_kernel[grid](arg_0, arg_1, arg_2, arg_3)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((10,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2, 10), (10, 1), device='cuda:0', dtype=torch.float32)
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
