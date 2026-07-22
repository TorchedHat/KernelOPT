import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused_native_layer_norm_view_30(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    # One block per row: x0 is a scalar row index
    x0 = tl.program_id(0)
    xmask = x0 < xnumel
    r0_index = tl.arange(0, R0_BLOCK)
    r0_mask = r0_index < r0_numel
    r0_1 = r0_index
    # Load: addresses are 1D shape [R0_BLOCK]
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768 * x0), r0_mask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    # Mean: reduce 1D vector over axis=0
    tmp6 = tl.where(r0_mask & xmask, tmp0, 0.0)
    tmp7 = tl.sum(tmp6, 0).to(tl.float32)
    tmp8 = tl.full([], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    # Variance: reduce 1D vector over axis=0
    tmp11 = tmp0 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.where(r0_mask & xmask, tmp12, 0.0)
    tmp16 = tl.sum(tmp13, 0).to(tl.float32)
    tmp17 = tmp0 - tmp10
    tmp18 = tl.full([], 768.0, tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tl.full([], 1e-05, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r0_1 + 768 * x0), tmp27, r0_mask & xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 1960
    r0_numel = 768
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_native_layer_norm_view_30[grid](arg_0, arg_1, arg_2, arg_3, 1960, 768, XBLOCK=128)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((10, 14, 14, 768), (150528, 10752, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
