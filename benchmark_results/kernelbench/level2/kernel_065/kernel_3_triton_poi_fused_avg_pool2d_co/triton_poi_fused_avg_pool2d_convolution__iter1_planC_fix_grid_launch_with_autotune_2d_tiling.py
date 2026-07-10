import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 8,  'YBLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 512},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 512},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 256},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 128},  num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64},   num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 2048}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 2048}, num_warps=8, num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_avg_pool2d_convolution_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1155200
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 95)
    y1 = ((yindex // 95) % 95)
    y2 = yindex // 9025
    y4 = (yindex % 9025)
    tmp0 = tl.load(in_ptr0 + (x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (64 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (192 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (24448 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (24512 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (24576 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (24640 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (48896 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (48960 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (49024 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (49088 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (73344 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (73408 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (73472 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (73536 + x3 + 256*y0 + 97792*y1 + 9339136*y2), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full([1, 1], 16.0, tl.float32)
    tmp32 = (tmp30 / tmp31)
    tl.store(out_ptr0 + (y4 + 9025*x3 + 577600*y2), tmp32, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 1155200
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_avg_pool2d_convolution_3[grid](arg_0, arg_1, 1155200, 64)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 64, 382, 382), (9339136, 1, 24448, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 95, 95), (577600, 9025, 95, 1), device='cuda:0', dtype=torch.float32)
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
