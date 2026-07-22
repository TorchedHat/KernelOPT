import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    NUM_SMS : tl.constexpr = 132

    M = 512
    N = 512
    K = 96

    stride_aq = 49152   # per-head stride for A (96*512)
    stride_am = 96
    stride_ak = 1
    stride_ab = 393216  # per-batch stride for A (8*49152)

    stride_bq = 49152   # per-head stride for B (96*512)
    stride_bk = 512
    stride_bn = 1
    stride_bb = 393216  # per-batch stride for B (8*49152)

    # Total tiles: 1024 batch_heads x 64 MN-tiles = 65536
    total_tiles = 65536

    sm_id = tl.program_id(0).to(INDEX_DTYPE)
    tile_id = sm_id

    while tile_id < total_tiles:
        # CRITICAL: reset acc at the start of EACH tile iteration
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        # Decompose tile_id into batch*head index and MN tile indices
        idx_q = tile_id // 64   # batch*head index 0..1023
        mn_tile = tile_id % 64
        pid_m = mn_tile // 8
        pid_n = mn_tile % 8

        # Compute row/col ranges
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        # Boundary-safe indices
        ram = rm % M
        rbn = rn % N

        # Batch and head indices
        batch_idx = idx_q // 8
        head_idx = idx_q % 8

        # Set up pointers for A and B
        A_ptr = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + batch_idx * stride_ab + head_idx * stride_aq)
        B_ptr = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + batch_idx * stride_bb + head_idx * stride_bq)

        # K-reduction loop
        for k in range(K, 0, -BLOCK_K):
            if EVEN_K:
                a = tl.load(A_ptr)
                b = tl.load(B_ptr)
            else:
                a = tl.load(A_ptr, mask=rk[None, :] < k, other=0.)
                b = tl.load(B_ptr, mask=rk[:, None] < k, other=0.)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_ptr += BLOCK_K * stride_ak
            B_ptr += BLOCK_K * stride_bk

        # Store result
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)
        xindex = idx_n + 512 * idx_m + 262144 * idx_q
        tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)

        tile_id += NUM_SMS

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (64, 1, 1)
    triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 8, 512, 96), (393216, 49152, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 8, 96, 512), (393216, 49152, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
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
