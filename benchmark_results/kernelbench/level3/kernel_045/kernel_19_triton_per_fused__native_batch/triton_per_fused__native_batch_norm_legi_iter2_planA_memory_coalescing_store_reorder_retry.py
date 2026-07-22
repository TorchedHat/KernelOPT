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
    # One CTA per (x0, x1) pair — coalesced access pattern
    pid = tl.program_id(0)
    x0 = pid % 512          # scalar channel index (stride=512 in output)
    x1 = pid // 512         # scalar batch*head index
    x3 = x1 % 8            # scalar head index
    x4 = x1 // 8           # scalar batch index
    r0_index = tl.arange(0, R0_BLOCK)  # [64] — stride-1 in both input and output
    # Load input: in_out_ptr0 has strides (262144, 1, 32768, 512)
    # index = x0 + 512*r0_index + 32768*x1 — r0_index is stride-1 in output (coalesced)
    tmp0 = tl.load(in_out_ptr0 + (x0 + 512*r0_index + 32768*x1), None)
    # Load per-channel BatchNorm parameters as scalars
    tmp1 = tl.load(in_ptr0 + x0, None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + x0, None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + x0, None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + x0, None, eviction_policy='evict_last')
    # BatchNorm computation
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp7 = tmp5 + 1e-05
    tmp8 = tl.sqrt_rn(tmp7)
    tmp12 = 1.0 / tmp8
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16  # [64] normalized + affine
    # Softmax over r0 dimension (64 elements)
    tmp22 = tl.max(tmp17, axis=0)   # scalar max
    tmp23 = tmp17 - tmp22
    tmp24 = tl.exp(tmp23)
    tmp27 = tl.sum(tmp24, axis=0)   # scalar sum
    tmp28 = tmp17 - tmp22
    tmp29 = tl.exp(tmp28)
    tmp30 = tmp29 / tmp27           # [64] softmax output
    # Store: out_ptr2 has strides (524288, 512, 64, 1)
    # index = r0_index + 64*x3 + 512*x0 + 524288*x4 — r0_index is stride-1 (coalesced)
    tl.store(out_ptr2 + (r0_index + 64*x3 + 512*x0 + 524288*x4), tmp30, None)

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
