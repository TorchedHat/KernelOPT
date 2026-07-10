import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_convolution_max_pool2d_with_indices_mul_tanh_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 508032
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 63)
    y1 = ((yindex // 63) % 63)
    y2 = yindex // 3969
    y4 = (yindex % 3969)
    tmp0 = tl.load(in_ptr0 + (x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (64 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (192 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (16256 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (16320 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (16384 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (16448 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (32512 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (32576 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (32640 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (32704 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (48768 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (48832 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (48896 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (48960 + x3 + 256*y0 + 65024*y1 + 4129024*y2), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp10 = tl.maximum(tmp8, tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tmp16 = tl.maximum(tmp14, tmp15)
    tmp18 = tl.maximum(tmp16, tmp17)
    tmp20 = tl.maximum(tmp18, tmp19)
    tmp22 = tl.maximum(tmp20, tmp21)
    tmp24 = tl.maximum(tmp22, tmp23)
    tmp26 = tl.maximum(tmp24, tmp25)
    tmp28 = tl.maximum(tmp26, tmp27)
    tmp30 = tl.maximum(tmp28, tmp29)
    tl.store(out_ptr0 + (y4 + 3969*x3 + 254016*y2), tmp30, xmask & ymask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 508032
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_add_convolution_max_pool2d_with_indices_mul_tanh_3[grid](arg_0, arg_1, 508032, 64, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 64, 254, 254), (4129024, 1, 16256, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 63, 63), (254016, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
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
