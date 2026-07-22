import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__softmax_bmm_exp_sub_transpose_view_2(arg_A, arg_B, out_ptr0):
    # Constants
    BATCH     : tl.constexpr = 2048
    M         : tl.constexpr = 32
    N         : tl.constexpr = 512
    HEAD_DIM  : tl.constexpr = 100
    BLOCK_M   : tl.constexpr = 32
    BLOCK_N   : tl.constexpr = 64
    BLOCK_K   : tl.constexpr = 32

    # Each CTA handles one (batch_idx, query_tile) pair
    pid    = tl.program_id(0)
    GRID_M = tl.cdiv(M, BLOCK_M)   # = 1  (M=32, BLOCK_M=32)
    batch_idx  = pid // GRID_M
    query_tile = pid  % GRID_M

    # Query row offsets for this tile
    rm = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    rm_mask = rm < M                                      # [BLOCK_M]

    # ----------------------------------------------------------------
    # PASS 1: stream over KV blocks to compute running_max & running_sum
    # ----------------------------------------------------------------
    running_max = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Q layout: (204800, 32) with strides (32, 1)
    #   logical shape [BATCH, M, HEAD_DIM] viewed as [BATCH*M, HEAD_DIM]
    #   Q[batch, m, k] = arg_A[batch*M + m, k]  (stride_am=1, stride_ak=32)
    # K layout: (2048, 100, 512) with strides (51200, 512, 1)
    #   K[batch, k, n] = arg_B[batch*51200 + k*512 + n]

    q_base = arg_A + batch_idx * M * HEAD_DIM  # base for this batch's Q rows
    k_base = arg_B + batch_idx * HEAD_DIM * N  # base for this batch's K

    for kv_start in tl.range(0, N, BLOCK_N):
        rn = kv_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
        rn_mask = rn < N                          # [BLOCK_N]

        # Compute scores[BLOCK_M, BLOCK_N] = Q[rm, :] @ K[:, rn]^T
        # = sum over k: Q[rm, k] * K[k, rn]
        # We do this by iterating over HEAD_DIM in BLOCK_K chunks
        scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        rk = tl.arange(0, BLOCK_K)
        for k_start in tl.static_range(0, HEAD_DIM, BLOCK_K):
            k_remaining = HEAD_DIM - k_start
            k_mask = rk < k_remaining

            # Q tile: [BLOCK_M, BLOCK_K]
            # Q[batch, m, k] stored as arg_A[batch*M + m, k] with strides (32, 1)
            # So offset = (batch_idx*M + rm)*HEAD_DIM + (k_start + rk)
            # But actual strides: arg_A has shape (204800, 32) stride (32, 1)
            # meaning arg_A[row, col] where row = batch*M+m, col = k
            # offset = row*32 + col*1 = (batch_idx*M + rm)*32 + (k_start+rk)
            q_off = (rm[:, None]) * 32 + (k_start + rk[None, :])  # [BLOCK_M, BLOCK_K]
            q_mask = rm_mask[:, None] & k_mask[None, :]
            q_tile = tl.load(q_base + q_off, mask=q_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

            # K tile: [BLOCK_K, BLOCK_N]  (K stored as [HEAD_DIM, N])
            # K[batch, k, n] = arg_B[batch*51200 + k*512 + n]
            k_off = (k_start + rk[:, None]) * N + rn[None, :]  # [BLOCK_K, BLOCK_N]
            k_mask2 = k_mask[:, None] & rn_mask[None, :]
            k_tile = tl.load(k_base + k_off, mask=k_mask2, other=0.0)  # [BLOCK_K, BLOCK_N]

            scores = tl.dot(q_tile, k_tile, acc=scores, allow_tf32=False)

        # Update running max and sum
        block_max = tl.max(scores, axis=1)                    # [BLOCK_M]
        new_max   = tl.maximum(running_max, block_max)        # [BLOCK_M]
        alpha     = tl.exp(running_max - new_max)             # [BLOCK_M]
        running_sum = running_sum * alpha + tl.sum(tl.exp(scores - new_max[:, None]), axis=1)
        running_max = new_max

    # ----------------------------------------------------------------
    # PASS 2: stream over KV blocks again, compute normalized softmax
    #         and write directly to output
    # ----------------------------------------------------------------
    for kv_start in tl.range(0, N, BLOCK_N):
        rn = kv_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
        rn_mask = rn < N

        # Recompute scores for this KV block
        scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        rk = tl.arange(0, BLOCK_K)
        for k_start in tl.static_range(0, HEAD_DIM, BLOCK_K):
            k_remaining = HEAD_DIM - k_start
            k_mask = rk < k_remaining

            q_off = (rm[:, None]) * 32 + (k_start + rk[None, :])
            q_mask = rm_mask[:, None] & k_mask[None, :]
            q_tile = tl.load(q_base + q_off, mask=q_mask, other=0.0)

            k_off = (k_start + rk[:, None]) * N + rn[None, :]
            k_mask2 = k_mask[:, None] & rn_mask[None, :]
            k_tile = tl.load(k_base + k_off, mask=k_mask2, other=0.0)

            scores = tl.dot(q_tile, k_tile, acc=scores, allow_tf32=False)

        # Normalize
        p = tl.exp(scores - running_max[:, None]) / running_sum[:, None]  # [BLOCK_M, BLOCK_N]

        # Store: out[batch, m, n] = out_ptr0[batch*16384 + m*512 + n]
        out_off = batch_idx * 16384 + rm[:, None] * 512 + rn[None, :]
        out_mask = rm_mask[:, None] & rn_mask[None, :]
        tl.store(out_ptr0 + out_off, p, mask=out_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (4, 1, 1)
    triton_tem_fused__softmax_bmm_exp_sub_transpose_view_2[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((204800, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 100, 512), (51200, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 32, 512), (16384, 512, 1), device='cuda:0', dtype=torch.float32)
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
