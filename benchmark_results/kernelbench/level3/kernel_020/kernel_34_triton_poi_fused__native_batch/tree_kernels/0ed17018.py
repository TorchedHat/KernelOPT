import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Swizzle: group GROUP_SIZE=8 blocks per channel stripe for L2 locality
    # num_blocks_per_stripe = ceil(960 / XBLOCK)
    num_blocks_per_stripe = (960 + XBLOCK - 1) // XBLOCK
    group_size_total = num_blocks_per_stripe * 8
    group_id = pid // group_size_total
    local_id = pid % group_size_total
    stripe_id = local_id % num_blocks_per_stripe
    outer_id = local_id // num_blocks_per_stripe
    swizzled_pid = group_id * group_size_total + stripe_id * 8 + outer_id
    xoffset = swizzled_pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 960)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp_inv_std = tl.rsqrt(tmp3 + 1e-5)
    tmp11 = tmp2 * tmp_inv_std
    tmp15 = libdevice.fma(tmp11, tmp12, tmp14)
    tmp17 = tl.maximum(tmp15, 0.0)
    tmp19 = tl.minimum(tmp17, 6.0)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 470400
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_34[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 470400)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((960,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4]

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
