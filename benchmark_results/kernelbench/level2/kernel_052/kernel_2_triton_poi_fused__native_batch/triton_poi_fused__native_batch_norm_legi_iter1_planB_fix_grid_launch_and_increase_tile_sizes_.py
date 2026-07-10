import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'YBLOCK': 32,  'XBLOCK': 128}, num_warps=4,  num_stages=1),
    triton.Config({'YBLOCK': 64,  'XBLOCK': 128}, num_warps=4,  num_stages=1),
    triton.Config({'YBLOCK': 128, 'XBLOCK': 128}, num_warps=8,  num_stages=1),
    triton.Config({'YBLOCK': 256, 'XBLOCK': 128}, num_warps=8,  num_stages=1),
    triton.Config({'YBLOCK': 512, 'XBLOCK': 128}, num_warps=16, num_stages=1),
    triton.Config({'YBLOCK': 64,  'XBLOCK': 64},  num_warps=4,  num_stages=1),
    triton.Config({'YBLOCK': 128, 'XBLOCK': 64},  num_warps=8,  num_stages=1),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mul_softplus_tanh_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1016064
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 15876)
    y1 = yindex // 15876
    tmp0 = tl.load(in_ptr0 + (x2 + 128*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 20.0, tl.float32)
    tmp4 = tmp2 > tmp3
    tmp5 = tl.exp(tmp2)
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = tmp8 * tmp2
    tmp11 = tmp9 - tmp10
    tmp13 = tl.full([1, 1], 1e-05, tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sqrt_rn(tmp14)
    tmp16 = tl.full([1, 1], 1, tl.int32)
    tmp17 = (tmp16 / tmp15)
    tmp18 = tl.full([1, 1], 1.0, tl.float32)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr0 + (y0 + 15876*x2 + 2032128*y1), tmp24, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    ynumel = 1016064
    xnumel = 128
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mul_softplus_tanh_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 1016064, 128)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((64, 128, 126, 126), (2032128, 1, 16128, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((64, 128, 126, 126), (2032128, 15876, 126, 1), device='cuda:0', dtype=torch.float32)
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
