import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54(arg_A, arg_B, out_ptr0):
    # A: shape (490, 768), strides (768, 1) -> A[m, k] = arg_A + m*768 + k
    # B: shape (3072, 768), strides (768, 1), accessed transposed -> B[k, n] = arg_B + k + n*768
    # out: shape (490, 3072), strides (3072, 1) -> out[m, n] = out_ptr0 + m*3072 + n
    M : tl.constexpr = 490
    N : tl.constexpr = 3072
    K : tl.constexpr = 768
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                     # [BLOCK_K]

    # A[m, k]: stride_am=768, stride_ak=1
    a_ptrs = arg_A + offs_m[:, None] * 768 + offs_k[None, :]
    # B[k, n]: stride_bk=1, stride_bn=768
    b_ptrs = arg_B + offs_k[:, None] + offs_n[None, :] * 768

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy='evict_first')
        b = tl.load(b_ptrs, mask=b_mask, other=0.0, eviction_policy='evict_last')
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
        a_ptrs += BLOCK_K      # advance K dim of A (stride_ak=1)
        b_ptrs += BLOCK_K      # advance K dim of B (stride_bk=1)

    # Store result
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr0 + offs_m[:, None] * 3072 + offs_n[None, :], acc, mask=mask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (192, 1, 1)
    triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 49, 768), (37632, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
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
