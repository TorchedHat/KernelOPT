import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused_mm_native_layer_norm_t_view_50(arg_A, arg_B, out_ptr0):
    ALLOW_TF32: tl.constexpr = False
    ACC_TYPE: tl.constexpr = tl.float32
    GROUP_M: tl.constexpr = 8
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 32
    M: tl.constexpr = 490
    N: tl.constexpr = 768
    K: tl.constexpr = 1536

    if M * N == 0:
        return

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    a_block_ptr = tl.make_block_ptr(
        base=arg_A,
        shape=(M, K),
        strides=(1536, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr0,
        shape=(M, N),
        strides=(768, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # Transposed B load: shape (BLOCK_N, BLOCK_K) with stride-1 along K dimension
    # This eliminates 12-way shared memory bank conflicts caused by stride-1536 access
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # b_ptrs shape: (BLOCK_N, BLOCK_K) -- consecutive threads vary along K (stride-1)
    b_ptrs = arg_B + offs_n[:, None] * 1536 + offs_k[None, :] * 1

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(tl.cdiv(K, BLOCK_K)):
        k_offset = k_idx * BLOCK_K
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        # b_mask: rows check offs_n < N, cols check offs_k + k_offset < K
        b_mask = (offs_n[:, None] < N) & ((offs_k[None, :] + k_offset) < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # (BLOCK_N, BLOCK_K)
        # tl.trans(b) presents b as (BLOCK_K, BLOCK_N) for the dot product
        acc += tl.dot(a, tl.trans(b), allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_ptrs += BLOCK_K  # advance along K dimension (stride-1 in memory)

    tl.store(out_block_ptr, acc, boundary_check=(0, 1))

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (96, 1, 1)
    triton_tem_fused_mm_native_layer_norm_t_view_50[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 49, 1536), (75264, 1536, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 768), (768, 1), device='cuda:0', dtype=torch.float32)
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
