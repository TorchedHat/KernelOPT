import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clamp_convolution_max_pool2d_with_indices_mul_native_group_norm_4(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 123008
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    # xindex = channel (0..63, stride=1 in input memory → coalesced)
    # yindex = batch * spatial (0..123007)
    x4 = yindex % 961   # output spatial index (0..960)
    y1 = yindex // 961  # batch index (0..127)
    x2 = x4 % 31        # output col
    x3 = x4 // 31       # output row
    # base pointer: channel (stride=1) + col*64 (stride=64 per col in input) +
    #               row*8064 (stride=8064 per row in input, but 32256//4=8064) +
    #               batch*1016064
    # Input strides: (1016064, 1, 8064, 64) for (N, C, H, W)
    # base = channel + 64*col + 8064*row + 1016064*batch
    #      = xindex + 256*x2 + 32256*x3 + 1016064*y1
    # (256 = 64*4 because pool window col stride is 64, and 32256 = 8064*4 because pool window row stride is 8064)
    # Wait: input col stride=64, so adjacent pool cols differ by 64. Pool window is 4x4 with stride 4.
    # x2 is output col (0..30), input col = x2*4, so col offset = x2*4*64 = 256*x2 ✓
    # x3 is output row (0..30), input row = x3*4, so row offset = x3*4*8064 = 32256*x3 ✓
    base = tl.multiple_of(xindex + 256*x2 + 32256*x3 + 1016064*y1, 64)
    tmp0 = tl.load(in_ptr0 + base, mask=ymask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + (64 + base), mask=ymask, eviction_policy='evict_first')
    tmp3 = tl.load(in_ptr0 + (128 + base), mask=ymask, eviction_policy='evict_first')
    tmp5 = tl.load(in_ptr0 + (192 + base), mask=ymask, eviction_policy='evict_first')
    tmp7 = tl.load(in_ptr0 + (8064 + base), mask=ymask, eviction_policy='evict_first')
    tmp9 = tl.load(in_ptr0 + (8128 + base), mask=ymask, eviction_policy='evict_first')
    tmp11 = tl.load(in_ptr0 + (8192 + base), mask=ymask, eviction_policy='evict_first')
    tmp13 = tl.load(in_ptr0 + (8256 + base), mask=ymask, eviction_policy='evict_first')
    tmp15 = tl.load(in_ptr0 + (16128 + base), mask=ymask, eviction_policy='evict_first')
    tmp17 = tl.load(in_ptr0 + (16192 + base), mask=ymask, eviction_policy='evict_first')
    tmp19 = tl.load(in_ptr0 + (16256 + base), mask=ymask, eviction_policy='evict_first')
    tmp21 = tl.load(in_ptr0 + (16320 + base), mask=ymask, eviction_policy='evict_first')
    tmp23 = tl.load(in_ptr0 + (24192 + base), mask=ymask, eviction_policy='evict_first')
    tmp25 = tl.load(in_ptr0 + (24256 + base), mask=ymask, eviction_policy='evict_first')
    tmp27 = tl.load(in_ptr0 + (24320 + base), mask=ymask, eviction_policy='evict_first')
    tmp29 = tl.load(in_ptr0 + (24384 + base), mask=ymask, eviction_policy='evict_first')
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
    # Output strides: (61504, 961, 31, 1) for (N, C, H, W)
    # out[n, c, h, w] = n*61504 + c*961 + h*31 + w
    #                 = y1*61504 + xindex*961 + x3*31 + x2
    #                 = y1*61504 + xindex*961 + x4
    tl.store(out_ptr1 + (xindex*961 + x4 + 61504*y1), tmp34, mask=ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 8192
    xnumel = 961
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_clamp_convolution_max_pool2d_with_indices_mul_native_group_norm_4[grid](arg_0, arg_1, 8192, 961, YBLOCK=8, XBLOCK=128)
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
