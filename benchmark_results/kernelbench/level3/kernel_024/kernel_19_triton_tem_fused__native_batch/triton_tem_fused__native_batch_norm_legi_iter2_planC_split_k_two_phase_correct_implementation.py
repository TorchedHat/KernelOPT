import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'SPLIT_K': 4,  'BLOCK_M': 2, 'BLOCK_N': 32,  'BLOCK_K': 64}, num_warps=2, num_stages=2),
    triton.Config({'SPLIT_K': 4,  'BLOCK_M': 2, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
    triton.Config({'SPLIT_K': 8,  'BLOCK_M': 2, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
    triton.Config({'SPLIT_K': 8,  'BLOCK_M': 2, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'SPLIT_K': 16, 'BLOCK_M': 2, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
    triton.Config({'SPLIT_K': 16, 'BLOCK_M': 2, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
], key=['M', 'N', 'K'])
@triton.jit
def split_k_gemm_kernel_19(arg_A, arg_B, ws_ptr, M, N, K,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid   = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m  = pid // grid_n
    pid_n  = pid  % grid_n
    k_start = pid_k * BLOCK_K
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = k_start + tl.arange(0, BLOCK_K)
    a_mask = (rm[:, None] < M) & (rk[None, :] < K)
    a = tl.load(arg_A + rm[:, None] * 216 + rk[None, :], mask=a_mask, other=0.0)
    b_mask = (rk[:, None] < K) & (rn[None, :] < N)
    b = tl.load(arg_B + rk[:, None] * 1 + rn[None, :] * 216, mask=b_mask, other=0.0)
    acc = tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.atomic_add(ws_ptr + rm[:, None] * N + rn[None, :], acc, mask=out_mask)


@triton.jit

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (27, 1, 1)
    split_k_gemm_kernel_19[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2, 216, 1, 1), (216, 1, 216, 216), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((864, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 864, 1, 1), (864, 1, 864, 864), device='cuda:0', dtype=torch.float32)
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
