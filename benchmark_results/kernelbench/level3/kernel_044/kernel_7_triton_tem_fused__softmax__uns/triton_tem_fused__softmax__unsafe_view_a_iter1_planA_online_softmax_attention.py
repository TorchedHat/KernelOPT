import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_7(arg_A, arg_B, out_ptr0):
    # arg_A: attention scores (128, 8, 512, 512), strides (2097152, 262144, 512, 1)
    # arg_B: V matrix         (128, 8, 512, 96),  strides (393216, 49152, 96, 1)
    # out_ptr0: output        (1024, 512, 96),     strides (49152, 96, 1)
    # Computes: out = softmax(arg_A, dim=-1) @ arg_B  (fused online softmax attention)

    SEQ_Q    : tl.constexpr = 512
    SEQ_K    : tl.constexpr = 512
    HEAD_DIM : tl.constexpr = 96
    HEADS    : tl.constexpr = 8
    BLOCK_M  : tl.constexpr = 64
    BLOCK_N  : tl.constexpr = 64
    # BLOCK_D must be power-of-2 >= HEAD_DIM; use 128 for HEAD_DIM=96
    BLOCK_D  : tl.constexpr = 128

    # arg_A strides
    stride_ab : tl.constexpr = 2097152
    stride_ah : tl.constexpr = 262144
    stride_am : tl.constexpr = 512
    stride_an : tl.constexpr = 1

    # arg_B (V) strides
    stride_vb : tl.constexpr = 393216
    stride_vh : tl.constexpr = 49152
    stride_vn : tl.constexpr = 96
    stride_vd : tl.constexpr = 1

    # out strides
    stride_ob : tl.constexpr = 49152
    stride_om : tl.constexpr = 96
    stride_od : tl.constexpr = 1

    pid_m  = tl.program_id(0)   # query tile index
    pid_bh = tl.program_id(1)   # batch*heads flat index (0..1023)

    pid_b = pid_bh // HEADS
    pid_h = pid_bh % HEADS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # BLOCK_D=128 is power-of-2; mask to HEAD_DIM=96 for correctness
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM

    # Online softmax state
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    A_base = arg_A + pid_b * stride_ab + pid_h * stride_ah
    V_base = arg_B + pid_b * stride_vb + pid_h * stride_vh

    num_kv_blocks : tl.constexpr = SEQ_K // BLOCK_N  # 512 // 64 = 8

    for j in tl.static_range(num_kv_blocks):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load attention scores: [BLOCK_M, BLOCK_N]
        # SEQ_Q=512 is multiple of BLOCK_M=64, SEQ_K=512 is multiple of BLOCK_N=64
        scores = tl.load(
            A_base + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
            mask=(offs_m[:, None] < SEQ_Q),
            other=float('-inf')
        )

        # Online softmax update
        m_j   = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])

        # Load V tile: [BLOCK_N, BLOCK_D]
        v = tl.load(
            V_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=(d_mask[None, :]),
            other=0.0
        )

        # Accumulate: rescale existing acc, add new contribution
        acc  = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=False)
        l_i  = l_i * alpha + tl.sum(p, axis=1)
        m_i  = m_new

    # Normalize
    out = acc / l_i[:, None]

    # Store output: [BLOCK_M, BLOCK_D], only HEAD_DIM=96 valid columns
    # out_ptr0 layout: (1024, 512, 96) with strides (49152, 96, 1)
    tl.store(
        out_ptr0 + pid_bh * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        out,
        mask=(offs_m[:, None] < SEQ_Q) & (d_mask[None, :])
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
