import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6(arg_A, arg_B, out_ptr0):
    # arg_A: attention scores (128, 8, 512, 512) with strides (2097152, 262144, 512, 1)
    #        flattened batch-head: (1024, 512, 512), stride_bh=262144, stride_m=512, stride_n=1
    # arg_B: V matrix (128, 8, 512, 96) with strides (393216, 49152, 96, 1)
    #        flattened batch-head: (1024, 512, 96), stride_bh=49152, stride_m=96, stride_n=1
    # out_ptr0: output (1024, 512, 96), stride_bh=49152, stride_m=96, stride_n=1

    BLOCK_M      : tl.constexpr = 64
    BLOCK_N      : tl.constexpr = 64
    SEQ_LEN      : tl.constexpr = 512
    HEAD_DIM     : tl.constexpr = 96
    HEAD_DIM_PAD : tl.constexpr = 128  # next power-of-2 >= 96 for tl.dot

    # Strides (flattened batch-head dimension)
    stride_score_bh : tl.constexpr = 262144  # SEQ_LEN * SEQ_LEN
    stride_score_m  : tl.constexpr = 512     # SEQ_LEN
    stride_score_n  : tl.constexpr = 1

    stride_v_bh : tl.constexpr = 49152       # SEQ_LEN * HEAD_DIM
    stride_v_m  : tl.constexpr = 96          # HEAD_DIM
    stride_v_n  : tl.constexpr = 1

    stride_out_bh : tl.constexpr = 49152     # SEQ_LEN * HEAD_DIM
    stride_out_m  : tl.constexpr = 96        # HEAD_DIM
    stride_out_n  : tl.constexpr = 1

    pid_m  = tl.program_id(0)  # query tile index along SEQ_LEN
    pid_bh = tl.program_id(1)  # batch-head index (0..1023)

    # Offsets for query rows and head dim
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PAD)

    # Masks
    m_mask = offs_m < SEQ_LEN
    d_mask = offs_d < HEAD_DIM

    # Online softmax state
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PAD], dtype=tl.float32)

    # Base pointers for this batch-head
    score_base = arg_A + pid_bh * stride_score_bh
    v_base     = arg_B + pid_bh * stride_v_bh

    # Stream over KV sequence dimension
    for start_n in tl.range(0, SEQ_LEN, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < SEQ_LEN

        # Load attention scores tile: [BLOCK_M, BLOCK_N]
        score_ptrs = score_base + offs_m[:, None] * stride_score_m + offs_n[None, :] * stride_score_n
        scores = tl.load(score_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=-float('inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        # Load V tile: [BLOCK_N, HEAD_DIM_PAD]
        v_ptrs = v_base + offs_n[:, None] * stride_v_m + offs_d[None, :] * stride_v_n
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # Accumulate: acc = acc * alpha + p @ v
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v.to(tl.float32), allow_tf32=False)

        # Update running sum and max
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output: only HEAD_DIM=96 columns
    out_base = out_ptr0 + pid_bh * stride_out_bh
    out_ptrs = out_base + offs_m[:, None] * stride_out_m + offs_d[None, :] * stride_out_n
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & d_mask[None, :])

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
