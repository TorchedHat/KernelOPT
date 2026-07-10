import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused__softmax_add_convolution_exp_mean_mul_prepare_softmax_online_sub_tanh_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 262144
    r0_numel = 64

    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    x0 = (xindex % 16384)
    x1 = xindex // 16384

    # Loop 1: compute online softmax statistics (m_i, d_i)
    m_i = tl.full([XBLOCK, 1], float('-inf'), tl.float32)
    d_i = tl.zeros([XBLOCK, 1], tl.float32)

    for r in tl.range(0, 64):
        r_idx = tl.full([XBLOCK, 1], r, tl.int32)
        val = tl.load(in_out_ptr0 + (x0 + 16384 * r_idx + 1048576 * x1))
        bias = tl.load(in_ptr0 + r_idx, eviction_policy='evict_last')
        score = val / 32.0 + bias
        m_new = tl.maximum(m_i, score)
        d_i = d_i * tl.exp(m_i - m_new) + tl.exp(score - m_new)
        m_i = m_new

    # Loop 2: normalize, apply tanh * 2.0, and store
    for r in tl.range(0, 64):
        r_idx = tl.full([XBLOCK, 1], r, tl.int32)
        val = tl.load(in_out_ptr0 + (x0 + 16384 * r_idx + 1048576 * x1))
        bias = tl.load(in_ptr0 + r_idx, eviction_policy='evict_last')
        score = val / 32.0 + bias
        softmax_val = tl.exp(score - m_i) / d_i
        out = libdevice.tanh(softmax_val) * 2.0
        tl.store(in_out_ptr0 + (x0 + 16384 * r_idx + 1048576 * x1), out)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 262144
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_add_convolution_exp_mean_mul_prepare_softmax_online_sub_tanh_1[grid](arg_0, arg_1, 262144, 64)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16, 64, 1, 128, 128), (1048576, 16384, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1]

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
