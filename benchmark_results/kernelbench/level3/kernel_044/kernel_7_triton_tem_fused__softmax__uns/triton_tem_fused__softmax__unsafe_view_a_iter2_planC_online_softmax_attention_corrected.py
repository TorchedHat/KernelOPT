import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_7(arg_A, arg_B, out_ptr0):
    # Flash Attention online softmax kernel
    # arg_A: scores (128, 8, 512, 512) strides=(2097152, 262144, 512, 1)
    # arg_B: V     (128, 8, 512, 96)  strides=(393216, 49152, 96, 1)
    # out_ptr0: output (1024, 512, 96) strides=(49152, 96, 1)

    BLOCK_M   : tl.constexpr = 64
    BLOCK_N   : tl.constexpr = 64
    HEAD_DIM  : tl.constexpr = 96
    HEAD_DIM_PADDED : tl.constexpr = 128  # pad to power-of-2 for tl.dot
    SEQ_LEN   : tl.constexpr = 512

    pid_m  = tl.program_id(0)   # query tile index (0..7)
    pid_bh = tl.program_id(1)   # batch*head index (0..1023)

    batch_idx = pid_bh // 8
    head_idx  = pid_bh % 8

    # Base pointers for this batch-head slice
    scores_base = arg_A + batch_idx * 2097152 + head_idx * 262144
    V_base      = arg_B + batch_idx * 393216  + head_idx * 49152
    out_base    = out_ptr0 + pid_bh * 49152

    # Query row offsets for this tile
    q_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_off < SEQ_LEN

    # Head dim offsets (for V and output)
    hd_off = tl.arange(0, HEAD_DIM_PADDED)
    hd_mask = hd_off < HEAD_DIM

    # Initialize running statistics
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

    # Stream over KV tiles
    for kv_start in range(0, SEQ_LEN, BLOCK_N):
        kv_off = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_off < SEQ_LEN

        # Load scores tile: [BLOCK_M, BLOCK_N]
        score_ptrs = scores_base + q_off[:, None] * 512 + kv_off[None, :]
        scores_tile = tl.load(
            score_ptrs,
            mask=q_mask[:, None] & kv_mask[None, :],
            other=float('-inf')
        )

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores_tile, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores_tile - m_new[:, None])  # [BLOCK_M, BLOCK_N]

        # Load V tile: [BLOCK_N, HEAD_DIM_PADDED] (padded)
        v_ptrs = V_base + kv_off[:, None] * 96 + hd_off[None, :]
        v_tile = tl.load(
            v_ptrs,
            mask=kv_mask[:, None] & hd_mask[None, :],
            other=0.0
        )  # [BLOCK_N, HEAD_DIM_PADDED]

        # Rescale accumulator and update
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v_tile.to(tl.float32))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Normalize
    out = acc / l_i[:, None]  # [BLOCK_M, HEAD_DIM_PADDED]

    # Store output (mask out padded head_dim columns)
    out_ptrs = out_base + q_off[:, None] * 96 + hd_off[None, :]
    tl.store(
        out_ptrs,
        out,
        mask=q_mask[:, None] & hd_mask[None, :]
    )

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (16, 1, 1)
    triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_7[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 8, 512, 512), (2097152, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 8, 512, 96), (393216, 49152, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024, 512, 96), (49152, 96, 1), device='cuda:0', dtype=torch.float32)
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
