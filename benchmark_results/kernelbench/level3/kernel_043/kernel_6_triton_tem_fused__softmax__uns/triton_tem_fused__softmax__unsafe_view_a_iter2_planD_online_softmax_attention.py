import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=[],
)
@triton.jit
def triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6(arg_A, arg_B, out_ptr0):
    # Fused online-softmax attention kernel
    # arg_A: attention scores (128, 8, 512, 512), strides (2097152, 262144, 512, 1)
    # arg_B: V matrix (128, 8, 512, 96), strides (393216, 49152, 96, 1)
    # out_ptr0: output (1024, 512, 96), strides (49152, 96, 1)

    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 64
    BLOCK_DV: tl.constexpr = 128   # next power of 2 >= 96

    SEQ_Q: tl.constexpr = 512
    SEQ_K: tl.constexpr = 512
    D_V: tl.constexpr = 96
    HEADS: tl.constexpr = 8

    # Strides for attention scores (128, 8, 512, 512)
    stride_ab: tl.constexpr = 2097152
    stride_ah: tl.constexpr = 262144
    stride_am: tl.constexpr = 512
    stride_an: tl.constexpr = 1

    # Strides for V (128, 8, 512, 96)
    stride_vb: tl.constexpr = 393216
    stride_vh: tl.constexpr = 49152
    stride_vn: tl.constexpr = 96
    stride_vd: tl.constexpr = 1

    # Strides for output (1024, 512, 96)
    stride_ob: tl.constexpr = 49152
    stride_om: tl.constexpr = 96
    stride_od: tl.constexpr = 1

    # Program IDs
    bh_idx = tl.program_id(0)   # batch * heads index: 0..1023
    m_idx  = tl.program_id(1)   # query tile index

    b_idx = bh_idx // HEADS
    h_idx = bh_idx % HEADS

    # Query tile offsets
    offs_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DV)

    # Base pointers for this batch/head
    A_base = arg_A + b_idx * stride_ab + h_idx * stride_ah
    B_base = arg_B + b_idx * stride_vb + h_idx * stride_vh

    # Online softmax state
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    # Mask for query rows
    mask_m = offs_m < SEQ_Q

    # Stream over KV blocks
    for n_start in tl.range(0, SEQ_K, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < SEQ_K

        # Load attention scores tile: [BLOCK_M, BLOCK_N]
        score_ptrs = A_base + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
        mask_mn = mask_m[:, None] & mask_n[None, :]
        scores = tl.load(score_ptrs, mask=mask_mn, other=float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V tile: [BLOCK_N, BLOCK_DV] (padded to BLOCK_DV=128, D_V=96)
        v_ptrs = B_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        mask_nd = mask_n[:, None] & (offs_d[None, :] < D_V)
        v_block = tl.load(v_ptrs, mask=mask_nd, other=0.0)

        # Accumulate weighted V sum (p: [BLOCK_M, BLOCK_N], v_block: [BLOCK_N, BLOCK_DV])
        acc = acc + tl.dot(p, v_block, allow_tf32=False)

        m_i = m_new

    # Finalize: divide by normalizer
    acc = acc / l_i[:, None]

    # Store output — only write valid D_V=96 columns
    out_ptrs = out_ptr0 + bh_idx * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    mask_out = mask_m[:, None] & (offs_d[None, :] < D_V)
    tl.store(out_ptrs, acc, mask=mask_out)

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
