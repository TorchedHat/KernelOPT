import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6(arg_A, arg_B, out_ptr0):
    # Online softmax attention kernel
    # arg_A: attention scores (128, 8, 512, 512), strides (2097152, 262144, 512, 1)
    # arg_B: V matrix (128, 8, 512, 96), strides (393216, 49152, 96, 1)
    # out_ptr0: output (1024, 512, 96), strides (49152, 96, 1)

    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    HEAD_DIM : tl.constexpr = 96
    HEAD_DIM_PADDED : tl.constexpr = 128  # next power of 2 >= 96
    SEQ_Q : tl.constexpr = 512
    SEQ_K : tl.constexpr = 512
    HEADS : tl.constexpr = 8

    # Strides for arg_A (scores): (batch, head, seq_q, seq_k)
    stride_ab : tl.constexpr = 2097152
    stride_ah : tl.constexpr = 262144
    stride_am : tl.constexpr = 512
    stride_an : tl.constexpr = 1

    # Strides for arg_B (V): (batch, head, seq_k, head_dim)
    stride_vb : tl.constexpr = 393216
    stride_vh : tl.constexpr = 49152
    stride_vn : tl.constexpr = 96
    stride_vd : tl.constexpr = 1

    # Strides for output: (batch*head, seq_q, head_dim)
    stride_ob : tl.constexpr = 49152
    stride_om : tl.constexpr = 96
    stride_od : tl.constexpr = 1

    pid_m = tl.program_id(0)  # query tile index
    pid_b = tl.program_id(1)  # combined batch*head index

    batch_idx = pid_b // HEADS
    head_idx = pid_b % HEADS

    # Query row offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Head dim offsets (padded to power of 2)
    offs_d = tl.arange(0, HEAD_DIM_PADDED)
    d_mask = offs_d < HEAD_DIM

    # Initialize online softmax state
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

    # Base pointer for scores: arg_A[batch_idx, head_idx, pid_m*BLOCK_M:, :]
    scores_base = arg_A + batch_idx * stride_ab + head_idx * stride_ah + offs_m[:, None] * stride_am

    # Base pointer for V: arg_B[batch_idx, head_idx, :, :]
    v_base = arg_B + batch_idx * stride_vb + head_idx * stride_vh

    # Stream over KV blocks
    for start_n in range(0, SEQ_K, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < SEQ_K

        # Load scores tile [BLOCK_M, BLOCK_N]
        m_mask = offs_m[:, None] < SEQ_Q
        scores_ptrs = scores_base + offs_n[None, :] * stride_an
        scores = tl.load(scores_ptrs,
                         mask=m_mask & n_mask[None, :],
                         other=float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        # Load V tile [BLOCK_N, HEAD_DIM_PADDED]
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs,
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0.0)

        # Accumulate: acc = acc * alpha + p @ V
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v.to(tl.float32), allow_tf32=False)

        # Update normalizer and max
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output [BLOCK_M, HEAD_DIM] to out_ptr0[pid_b, offs_m, offs_d]
    out_mask = (offs_m[:, None] < SEQ_Q) & d_mask[None, :]
    out_ptrs = out_ptr0 + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)

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
