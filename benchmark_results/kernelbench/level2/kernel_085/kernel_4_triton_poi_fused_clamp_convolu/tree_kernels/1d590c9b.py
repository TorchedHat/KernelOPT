import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_clamp_convolution_max_pool2d_with_indices_mul_native_group_norm_4(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 961
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 31)
    x3 = xindex // 31
    y0 = (yindex % 64)
    y1 = yindex // 64
    x4 = xindex
    y5 = yindex
    mask2d = ymask & xmask
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (64 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (192 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (8064 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (8128 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (8192 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (8256 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (16128 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (16192 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (16256 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (16320 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (24192 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (24256 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (24320 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (24384 + y0 + 256*x2 + 32256*x3 + 1016064*y1), mask=mask2d, eviction_policy='evict_last')
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
    tmp31 = tl.full([1, 1], 0.0, tl.float32)
    tmp32 = tl.maximum(tmp30, tmp31)
    tmp33 = tl.full([1, 1], 1.0, tl.float32)
    tmp34 = tl.minimum(tmp32, tmp33)
    tl.store(out_ptr1 + (x4 + 961*y5), tmp34, mask=mask2d)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 8192
    xnumel = 961
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_clamp_convolution_max_pool2d_with_indices_mul_native_group_norm_4[grid](arg_0, arg_1, 8192, 961)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 64, 126, 126), (1016064, 1, 8064, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 31, 31), (61504, 961, 31, 1), device='cuda:0', dtype=torch.float32)
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
