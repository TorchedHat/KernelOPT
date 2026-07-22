import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6(arg_A, arg_B, out_ptr0):
    # arg_A: attention scores (128, 8, 512, 512), strides (2097152, 262144, 512, 1)
    # arg_B: V               (128, 8, 512, 96),  strides (393216, 49152, 96, 1)
    # out_ptr0: output       (1024, 512, 96),     strides (49152, 96, 1)
    #
    # Fused online-softmax attention kernel (Flash Attention style).
    # Grid: (batch*num_heads, cdiv(seq_q, BLOCK_M), 1) = (1024, 8, 1)
    #
    # Hardcoded shapes and strides from original kernel:
    #   Scores: batch=128, num_heads=8, seq_q=512, seq_k=512
    #   V:      batch=128, num_heads=8, seq_k=512, head_dim_v=96
    #   Out:    (1024, 512, 96)

    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 64
    BLOCK_DMODEL_V: tl.constexpr = 96

    # Strides for Scores (arg_A): (2097152, 262144, 512, 1)
    stride_sb: tl.constexpr = 2097152
    stride_sh: tl.constexpr = 262144
    stride_sm: tl.constexpr = 512
    stride_sn: tl.constexpr = 1

    # Strides for V (arg_B): (393216, 49152, 96, 1)
    stride_vb: tl.constexpr = 393216
    stride_vh: tl.constexpr = 49152
    stride_vn: tl.constexpr = 96
    stride_vk: tl.constexpr = 1

    # Strides for Out (out_ptr0): (49152, 96, 1) — indexed as (batch*head, seq_q, head_dim_v)
    # We treat out as (1024, 512, 96): stride_ob_flat=49152, stride_om=96, stride_ok=1
    stride_ob_flat: tl.constexpr = 49152
    stride_om: tl.constexpr = 96
    stride_ok: tl.constexpr = 1

    num_heads: tl.constexpr = 8
    seq_q: tl.constexpr = 512
    seq_k: tl.constexpr = 512

    # Program IDs
    pid_bh = tl.program_id(0)  # batch*head index [0, 1024)
    pid_m = tl.program_id(1)   # Q tile index [0, 8)

    batch_id = pid_bh // num_heads
    head_id = pid_bh % num_heads

    # Row indices for this Q tile
    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offs < seq_q

    # Online softmax running state (float32 for numerical stability)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=tl.float32)

    # Base pointer for scores: batch_id * stride_sb + head_id * stride_sh + row * stride_sm
    scores_base = arg_A + batch_id * stride_sb + head_id * stride_sh + row_offs[:, None] * stride_sm

    # Base pointer for V: batch_id * stride_vb + head_id * stride_vh
    v_base = arg_B + batch_id * stride_vb + head_id * stride_vh

    # Stream over KV blocks
    for start_n in tl.range(0, seq_k, BLOCK_N):
        col_offs = start_n + tl.arange(0, BLOCK_N)
        col_mask = col_offs < seq_k

        # Load score tile [BLOCK_M, BLOCK_N]
        score_mask = row_mask[:, None] & col_mask[None, :]
        scores = tl.load(
            scores_base + col_offs[None, :] * stride_sn,
            mask=score_mask,
            other=float('-inf'),
        )

        # Online softmax update
        m_ij = tl.max(scores, axis=1)       # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)       # [BLOCK_M]
        alpha = tl.exp(m_i - m_new)         # rescale factor for existing acc
        p = tl.exp(scores - m_new[:, None]) # [BLOCK_M, BLOCK_N]

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc

        # Load V tile [BLOCK_N, BLOCK_DMODEL_V]
        V_blk = tl.load(
            v_base + col_offs[:, None] * stride_vn + tl.arange(0, BLOCK_DMODEL_V)[None, :] * stride_vk,
            mask=col_mask[:, None],
            other=0.0,
        )

        # Accumulate: p @ V_blk, keep in float32
        acc += tl.dot(p, V_blk.to(tl.float32), allow_tf32=False)

        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output [BLOCK_M, BLOCK_DMODEL_V] into out_ptr0 (1024, 512, 96)
    # out index: pid_bh * stride_ob_flat + row * stride_om + col * stride_ok
    out_base = out_ptr0 + pid_bh * stride_ob_flat + row_offs[:, None] * stride_om
    out_mask = row_mask[:, None]
    tl.store(
        out_base + tl.arange(0, BLOCK_DMODEL_V)[None, :] * stride_ok,
        acc,
        mask=out_mask,
    )

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (16, 1, 1)
    triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6[grid](arg_0, arg_1, arg_2)
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
