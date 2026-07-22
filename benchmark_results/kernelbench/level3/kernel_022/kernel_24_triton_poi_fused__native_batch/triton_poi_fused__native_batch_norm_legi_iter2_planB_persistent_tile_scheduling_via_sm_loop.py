import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    # Persistent kernel: 132 CTAs (one per SM on H200), each with XBLOCK=512 threads (16 warps)
    # Each CTA loops over its assigned tiles with stride = NUM_SMS * XBLOCK
    NUM_SMS = 132  # H200 has 132 SMs
    pid = tl.program_id(0)          # 0..NUM_SMS-1
    stride = NUM_SMS * XBLOCK       # step between tiles for this CTA
    xoffset = pid * XBLOCK          # this CTA's starting offset
    while xoffset < xnumel:
        xindex = xoffset + tl.arange(0, XBLOCK)
        xmask = xindex < xnumel
        x0 = xindex % 80
        tmp0  = tl.load(in_out_ptr0 + xindex, xmask)
        tmp1  = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
        tmp3  = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
        tmp2  = tmp0 - tmp1
        tmp4  = tl.full([1], 1e-05, tl.float32)
        tmp5  = tmp3 + tmp4
        tmp6  = tl.sqrt_rn(tmp5)
        tmp7  = tl.full([1], 1.0, tl.float32)
        tmp8  = tmp7 / tmp6
        tmp9  = tl.full([1], 1.0, tl.float32)
        tmp10 = tmp8 * tmp9
        tmp11 = tmp2 * tmp10
        tmp13 = tmp11 * tmp12
        tmp15 = tmp13 + tmp14
        tl.store(in_out_ptr0 + xindex, tmp15, xmask)
        xoffset += stride

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 156800
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 156800, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((80,), (1,), device='cuda:0', dtype=torch.float32)
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
