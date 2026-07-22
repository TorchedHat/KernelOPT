import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 961
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 31)
    x3 = xindex // 31
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (2*x2 + 126*x3 + 4000*y4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 126*x3 + 4000*y4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 2*x2 + 126*x3 + 4000*y4), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (63 + 2*x2 + 126*x3 + 4000*y4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (64 + 2*x2 + 126*x3 + 4000*y4), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (65 + 2*x2 + 126*x3 + 4000*y4), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (126 + 2*x2 + 126*x3 + 4000*y4), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (127 + 2*x2 + 126*x3 + 4000*y4), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (128 + 2*x2 + 126*x3 + 4000*y4), xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp10 = tl.maximum(tmp8, tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tmp16 = tl.maximum(tmp14, tmp15)
    tl.store(out_ptr0 + (y0 + 512*x5 + 492032*y1), tmp16, xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 32768
    xnumel = 961
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_cat_convolution_max_pool2d_with_indices_relu_31[grid](arg_0, arg_1, 32768, 961, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((64, 512, 63, 63), (2048000, 4000, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 512, 31, 31), (492032, 1, 15872, 512), device='cuda:0', dtype=torch.float32)
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
