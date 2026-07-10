import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=8),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_add_convolution_mul_native_group_norm_sigmoid_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64516
    yoffset = tl.program_id(1) * YBLOCK
    # Apply hints to 1D range before reshape to avoid shape mismatch
    yrange_1d = tl.max_contiguous(tl.multiple_of(yoffset + tl.arange(0, YBLOCK), YBLOCK), YBLOCK)
    yindex = yrange_1d[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    # Apply hints to 1D range before reshape
    xrange_1d = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xindex = xrange_1d[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    # y0 cycles 0..31 contiguously; hint enables vectorized 1D input loads
    y0 = tl.max_contiguous(yrange_1d % 32, 32)[:, None]
    y1 = yindex // 32
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32*x2 + 2064512*y1), xmask & ymask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y3 // 4), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y3 // 4), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.sigmoid(tmp6)
    tmp11 = tl.full([1, 1], 258064.0, tl.float32)
    tmp12 = (tmp10 / tmp11)
    tmp13 = tl.full([1, 1], 1e-05, tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    # FMA 1: fuse (tmp7 - tmp8) * tmp15 = tmp7*tmp15 + (-tmp8*tmp15)
    tmp16 = libdevice.fma(tmp7, tmp15, -tmp8 * tmp15)
    # FMA 2: fuse tmp16*tmp17 + tmp19 into single FFMA
    tmp20 = libdevice.fma(tmp16, tmp17, tmp19)
    tl.store(out_ptr0 + (x2 + 64516*y3), tmp20, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    arg_7 = inputs[7]
    arg_8 = inputs[8]
    ynumel = 4096
    xnumel = 64516
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_add_convolution_mul_native_group_norm_sigmoid_3[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 4096, 64516)
    return arg_8

def get_inputs():
    arg_0 = rand_strided((128, 32, 254, 254), (2064512, 1, 8128, 32), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128, 8, 1, 1), (8, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128, 8, 1, 1), (8, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((128, 32, 254, 254), (2064512, 64516, 254, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8]

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
