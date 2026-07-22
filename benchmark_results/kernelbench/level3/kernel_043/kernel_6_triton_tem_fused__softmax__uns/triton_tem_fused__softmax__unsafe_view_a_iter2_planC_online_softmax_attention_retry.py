import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6(arg_A, arg_B, out_ptr0):
    # Fused online-softmax attention kernel
    # arg_A: attention scores (128, 8, 512, 512) strides (2097152, 262144, 512, 1)
    # arg_B: V matrix (128, 8, 512, 96) strides (393216, 49152, 96, 1)
    # out_ptr0: output (1024, 512, 96) strides (49152, 96, 1)
    BLOCK_M: tl.constexpr = 64   # query tile size
    BLOCK_N: tl.constexpr = 64   # key/value tile size (streaming over seq_len=512)
    BLOCK_D: tl.constexpr = 128  # head dim padded to power-of-2 (actual D=96)

    # Grid: (ceil(M/BLOCK_M), H, B) = (8, 8, 128)
    pid_m = tl.program_id(0)   # query tile index
    pid_h = tl.program_id(1)   # head index
    pid_b = tl.program_id(2)   # batch index

    # Dimensions
    M = 512   # sequence length (query)
    N = 512   # sequence length (key/value)
    D = 96    # head dimension

    # Strides for arg_A (attention scores): (128, 8, 512, 512)
    stride_ab = 2097152  # batch stride
    stride_ah = 262144   # head stride
    stride_am = 512      # query row stride
    stride_an = 1        # key col stride

    # Strides for arg_B (V matrix): (128, 8, 512, 96)
    stride_vb = 393216   # batch stride
    stride_vh = 49152    # head stride
    stride_vn = 96       # seq stride
    stride_vd = 1        # head_dim stride

    # Strides for out_ptr0: (1024, 512, 96) -> (49152, 96, 1)
    # batch-head index = pid_b * 8 + pid_h
    stride_ob = 49152    # per batch-head stride (H*M*D = 8*512*96 = 393216 / 8 = 49152 per head)
    stride_om = 96       # query row stride
    stride_od = 1        # head_dim stride

    # Query row offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    # Base pointer for attention scores for this (batch, head)
    scores_base = arg_A + pid_b * stride_ab + pid_h * stride_ah

    # Base pointer for V for this (batch, head)
    v_base = arg_B + pid_b * stride_vb + pid_h * stride_vh

    # Initialize online softmax state
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Stream over KV blocks
    for start_n in range(0, N, BLOCK_N):
        cur_offs_n = start_n + offs_n
        n_mask = cur_offs_n < N

        # Load attention score tile: shape [BLOCK_M, BLOCK_N]
        score_ptrs = scores_base + offs_m[:, None] * stride_am + cur_offs_n[None, :] * stride_an
        scores = tl.load(score_ptrs,
                         mask=(offs_m[:, None] < M) & (n_mask[None, :]),
                         other=float('-inf'))

        # Online softmax recurrence (Flash Attention)
        # Step 1: compute new per-row max
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))  # [BLOCK_M], per-row max

        # Step 2: compute rescale factor for existing acc and l_i
        alpha = tl.exp(m_i - m_new)  # [BLOCK_M]

        # Step 3: compute softmax numerators for this tile
        p = tl.exp(scores - m_new[:, None])  # [BLOCK_M, BLOCK_N]

        # Step 4: rescale accumulator and normalizer BEFORE adding new contributions
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Step 5: update running max
        m_i = m_new

        # Step 6: load V tile and accumulate weighted sum
        v_ptrs = v_base + cur_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs,
                    mask=(n_mask[:, None]) & (offs_d[None, :] < D),
                    other=0.0)

        # p: [BLOCK_M, BLOCK_N], v: [BLOCK_N, BLOCK_D]
        acc += tl.dot(p, v)

    # Normalize by the running sum
    acc = acc / l_i[:, None]

    # Write output
    # out_ptr0 layout: (1024, 512, 96) with strides (49152, 96, 1)
    # batch-head index in the flattened 1024 dimension
    bh_idx = pid_b * 8 + pid_h
    out_ptrs = out_ptr0 + bh_idx * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs,
             acc.to(tl.float32),
             mask=(offs_m[:, None] < M) & (offs_d[None, :] < D))

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
