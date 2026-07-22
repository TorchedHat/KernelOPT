import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_tem_fused__softmax__unsafe_view_addmm_bmm_clone_eq_exp_masked_fill_mul_slice_split_sub_transpose_view_6(arg_A, arg_B, out_ptr0):
    # Online softmax attention kernel
    # arg_A: Scores tensor (128,8,512,512) with strides (2097152, 262144, 512, 1)
    # arg_B: V tensor (128,8,512,96) with strides (393216, 49152, 96, 1)
    # out_ptr0: Out tensor (1024,512,96) with strides (49152, 96, 1)
    #
    # Grid: (cdiv(SEQ_Q, BLOCK_M), BH, 1) = (8, 1024, 1)
    # pid_m = query tile, pid_b = batch*head index

    SEQ_Q: tl.constexpr = 512
    SEQ_K: tl.constexpr = 512
    HEAD_DIM: tl.constexpr = 96
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 64
    BLOCK_D: tl.constexpr = 128  # next power of 2 >= 96

    # Strides for Scores (128,8,512,512): viewed as (1024,512,512)
    stride_sb: tl.constexpr = 262144   # stride over batch*head dim (= 512*512)
    stride_sq: tl.constexpr = 512      # stride over query dim
    stride_sk: tl.constexpr = 1        # stride over key dim

    # Strides for V (128,8,512,96): viewed as (1024,512,96)
    stride_vb: tl.constexpr = 49152    # stride over batch*head dim (= 512*96)
    stride_vk: tl.constexpr = 96       # stride over key dim
    stride_vd: tl.constexpr = 1        # stride over head_dim

    # Strides for Out (1024,512,96)
    stride_ob: tl.constexpr = 49152    # stride over batch*head dim
    stride_oq: tl.constexpr = 96       # stride over query dim
    stride_od: tl.constexpr = 1        # stride over head_dim

    pid_m = tl.program_id(0)  # query tile index
    pid_b = tl.program_id(1)  # batch*head index (0..1023)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Base pointers for this batch*head slice
    Scores_ptr = arg_A + pid_b * stride_sb + offs_m[:, None] * stride_sq
    V_ptr = arg_B + pid_b * stride_vb
    Out_ptr = out_ptr0 + pid_b * stride_ob + offs_m[:, None] * stride_oq

    # Running online softmax state
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # running normalizer
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)       # running weighted sum

    # Stream over KV sequence in BLOCK_N tiles
    for start_n in range(0, SEQ_K, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load score tile [BLOCK_M, BLOCK_N]
        s = tl.load(
            Scores_ptr + offs_n[None, :] * stride_sk,
            mask=(offs_m[:, None] < SEQ_Q) & (offs_n[None, :] < SEQ_K),
            other=float('-inf'),
        )

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        m_i = m_new

        # Load V tile [BLOCK_N, BLOCK_D]
        v = tl.load(
            V_ptr + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd,
            mask=(offs_n[:, None] < SEQ_K) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        # Accumulate: rescale old acc and add new contribution
        acc = alpha[:, None] * acc + tl.dot(p, v, allow_tf32=False)

    # Normalize and store output
    acc = acc / l_i[:, None]
    tl.store(
        Out_ptr + offs_d[None, :] * stride_od,
        acc,
        mask=(offs_m[:, None] < SEQ_Q) & (offs_d[None, :] < HEAD_DIM),
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
