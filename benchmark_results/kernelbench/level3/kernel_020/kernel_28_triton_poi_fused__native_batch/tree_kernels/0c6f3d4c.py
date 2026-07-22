import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1128960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 576)
    # Preload all 576 BN channel parameters upfront as coalesced loads
    # to eliminate per-element non-coalesced L1TEX stalls
    chan_idx = tl.arange(0, 1024)
    chan_mask = chan_idx < 576
    smem_mean   = tl.load(in_ptr0 + chan_idx, mask=chan_mask, other=0.0)
    smem_var    = tl.load(in_ptr1 + chan_idx, mask=chan_mask, other=0.0)
    smem_weight = tl.load(in_ptr2 + chan_idx, mask=chan_mask, other=0.0)
    smem_bias   = tl.load(in_ptr3 + chan_idx, mask=chan_mask, other=0.0)
    tmp0 = tl.load(in_out_ptr0 + (x2), mask=xmask)
    tmp1  = tl.gather(smem_mean,   x0, 0)
    tmp3  = tl.gather(smem_var,    x0, 0)
    tmp12 = tl.gather(smem_weight, x0, 0)
    tmp14 = tl.gather(smem_bias,   x0, 0)
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 + 1e-5
    tmp6 = libdevice.rsqrt(tmp5)
    tmp11 = tmp2 * tmp6
    tmp15 = tmp11 * tmp12 + tmp14
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.maximum(tmp15, tmp16)
    tmp18 = tl.full([1], 6.0, tl.float32)
    tmp19 = tl.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 1128960
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_28[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1128960, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((576,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((576,), (1,), device='cuda:0', dtype=torch.float32)
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
