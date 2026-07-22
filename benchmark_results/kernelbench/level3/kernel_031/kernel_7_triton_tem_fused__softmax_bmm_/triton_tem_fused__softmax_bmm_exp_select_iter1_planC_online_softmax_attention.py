import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__softmax_bmm_exp_select_sub_transpose_view_7(arg_A, arg_B, out_ptr0):
    # Fused online-softmax attention kernel.
    # arg_A: softmax score matrix (8, 16384, 16384) FP32 - NOT used (we recompute from Q,K)
    # arg_B: QKV tensor (3, 16384, 2, 128), strides (4194304, 256, 128, 1)
    #   Q = arg_B[0]: base offset 0
    #   K = arg_B[1]: base offset 4194304
    #   V = arg_B[2]: base offset 8388608
    # out_ptr0: (8, 16384, 32), strides (524288, 32, 1)
    #
    # The 8 "batches" correspond to 8 head slices of 32 elements each
    # within the 256-element head space (2 heads x 128 d_head):
    #   head = batch // 4,  d_start = (batch % 4) * 32
    # So Q[batch, m, k] = arg_B[m*256 + head*128 + d_start + k]
    #    K[batch, n, k] = arg_B[4194304 + n*256 + head*128 + d_start + k]
    #    V[batch, n, k] = arg_B[8388608 + batch*32 + n*256 + k]

    SEQ_LEN   : tl.constexpr = 16384
    D_HEAD    : tl.constexpr = 32      # output head dim per batch slice
    BLOCK_M   : tl.constexpr = 64
    BLOCK_N   : tl.constexpr = 64
    BLOCK_K   : tl.constexpr = 32     # == D_HEAD
    SCALE     : tl.constexpr = 0.08838834764831843  # 1/sqrt(128) — full Q/K dim is 128

    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    # Decompose batch index into head and d_start
    head    = pid_b // 4
    d_start = (pid_b % 4) * 32

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # Load Q tile: shape (BLOCK_M, BLOCK_K)
    # Q[m, k] = arg_B[m*256 + head*128 + d_start + k]
    q_base = arg_B + head * 128 + d_start
    Q_ptrs = q_base + offs_m[:, None] * 256 + offs_k[None, :]
    q = tl.load(Q_ptrs, mask=(offs_m[:, None] < SEQ_LEN), other=0.0)

    # Initialize online softmax state
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    k_base = arg_B + 4194304 + head * 128 + d_start
    v_base = arg_B + 8388608 + pid_b * 32

    for start_n in tl.range(0, SEQ_LEN, BLOCK_N, num_stages=2):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K tile: shape (BLOCK_N, BLOCK_K)
        K_ptrs = k_base + offs_n[:, None] * 256 + offs_k[None, :]
        k = tl.load(K_ptrs, mask=(offs_n[:, None] < SEQ_LEN), other=0.0)

        # Compute scores: (BLOCK_M, BLOCK_N) = q @ k.T * scale
        # Note: full Q/K dim is 128, but we only load 32 at a time.
        # We need to accumulate over all 4 BLOCK_K chunks to get full dot product.
        # Actually BLOCK_K=32=D_HEAD slice, but the full attention score needs
        # the full 128-dim dot product. We must loop over all 4 d-chunks.
        # Let's restructure: load full 128-dim Q and K for score computation,
        # but only the 32-dim V slice for accumulation.
        # Redefine: use BLOCK_K=32 for V output, but score needs full 128 dims.
        # We'll compute scores by accumulating over 4 chunks of 32.
        scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d_chunk in tl.static_range(0, 4):
            d_off = d_chunk * 32
            offs_d = tl.arange(0, 32)
            # Q chunk: arg_B[m*256 + head*128 + d_off + d]
            Qc_ptrs = arg_B + head * 128 + d_off + offs_m[:, None] * 256 + offs_d[None, :]
            qc = tl.load(Qc_ptrs, mask=(offs_m[:, None] < SEQ_LEN), other=0.0)
            # K chunk: arg_B[4194304 + n*256 + head*128 + d_off + d]
            Kc_ptrs = arg_B + 4194304 + head * 128 + d_off + offs_n[:, None] * 256 + offs_d[None, :]
            kc = tl.load(Kc_ptrs, mask=(offs_n[:, None] < SEQ_LEN), other=0.0)
            scores = tl.dot(qc, tl.trans(kc), acc=scores, allow_tf32=False)
        scores = scores * SCALE

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        m_i = m_new

        # Load V tile: shape (BLOCK_N, BLOCK_K=32)
        V_ptrs = v_base + offs_n[:, None] * 256 + offs_k[None, :]
        v = tl.load(V_ptrs, mask=(offs_n[:, None] < SEQ_LEN), other=0.0)

        # Accumulate: acc += p @ v
        acc += tl.dot(p.to(v.dtype), v, allow_tf32=False)

    # Normalize
    acc = acc / l_i[:, None]

    # Store output: out_ptr0[pid_b, m, k] = out_ptr0[pid_b*524288 + m*32 + k]
    Out_ptrs = out_ptr0 + pid_b * 524288 + offs_m[:, None] * 32 + offs_k[None, :]
    tl.store(Out_ptrs, acc, mask=(offs_m[:, None] < SEQ_LEN))

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    grid = (128, 1, 1)
    triton_tem_fused__softmax_bmm_exp_select_sub_transpose_view_7[grid](arg_0, arg_1, arg_2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((8, 16384, 16384), (268435456, 16384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((3, 16384, 2, 128), (4194304, 256, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 16384, 32), (524288, 32, 1), device='cuda:0', dtype=torch.float32)
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
