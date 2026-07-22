import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 512)
    x1 = xindex // 512
    x5 = xindex
    x3 = ((xindex // 512) % 8)
    x4 = xindex // 4096
    tmp0 = tl.load(in_out_ptr0 + (x0 + 512*r0_2 + 32768*x1), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    # tmp17 has shape [XBLOCK, R0_BLOCK] after broadcast_to in original
    # but actually tmp17 is [XBLOCK, 1] since all inputs are [XBLOCK, 1]
    # The original broadcast_to makes all R0_BLOCK columns identical
    # Online softmax: since all columns of tmp17 are identical (same value per row),
    # we compute softmax over R0_BLOCK identical values = 1/R0_BLOCK for each element
    # But we implement it correctly using the broadcast shape for generality
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    # Online single-pass softmax over the R0_BLOCK dimension
    # Initialize running max and normalizer
    m_i = tl.full([XBLOCK, 1], float('-inf'), tl.float32)
    d_i = tl.full([XBLOCK, 1], 0.0, tl.float32)
    for j in tl.static_range(R0_BLOCK):
        # Extract column j: shape [XBLOCK, 1]
        x_j = tl.reshape(tl.broadcast_to(tl.expand_dims(tmp18[:, j], 1), [XBLOCK, 1]), [XBLOCK, 1])
        m_new = tl.maximum(m_i, x_j)
        d_i = d_i * tl.exp(m_i - m_new) + tl.exp(x_j - m_new)
        m_i = m_new
    # Compute final softmax output
    out_val = tl.exp(tmp18 - m_i) / d_i
    tl.store(out_ptr2 + (r0_2 + 64*x3 + 512*x0 + 524288*x4), out_val, None)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 32768
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_19[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 32768, 64, XBLOCK=128)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((8, 512, 8, 64), (262144, 1, 32768, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((8, 512, 8, 64), (524288, 512, 64, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6]

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
