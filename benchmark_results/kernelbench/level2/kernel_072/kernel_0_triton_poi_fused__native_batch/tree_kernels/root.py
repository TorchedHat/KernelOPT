import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256048128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = ((xindex // 250047) % 16)
    x0 = (xindex % 3969)
    x5 = xindex // 3969
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x0 + 4000*x5), tmp17, xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 256048128
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 256048128, XBLOCK=128)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((64, 16, 63, 63, 63), (4000752, 250047, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((64, 16, 63, 63, 63), (4032000, 252000, 4000, 63, 1), device='cuda:0', dtype=torch.float32)
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
