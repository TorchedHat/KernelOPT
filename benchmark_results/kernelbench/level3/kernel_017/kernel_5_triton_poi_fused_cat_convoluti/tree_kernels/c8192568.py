import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # Existing num_stages=2 configs (keep all)
        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=2, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=2, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=2, maxnreg=64),
        # New num_stages=3 configs to hide 12.1-cycle L1TEX stall
        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=2, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=2, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=3, maxnreg=64),
        # New num_stages=4 configs for maximum pipeline depth
        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=2, num_stages=4, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=4, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=4, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=4, maxnreg=64),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=2, num_stages=4, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=4, maxnreg=64),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_cat_convolution_relu_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yoffset + tl.arange(0, YBLOCK) < ynumel
    ymask = ymask[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    _xrange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xindex = _xrange[None, :]
    xmask = xoffset + tl.arange(0, XBLOCK) < xnumel
    xmask = xmask[None, :]
    y0 = (yindex % 128)
    x2 = xindex
    y1 = yindex // 128
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x2 + 4194304*y1 + (y0)), tmp4 & ymask & xmask, other=0.0, eviction_policy='evict_first')
    tmp7 = tl.maximum(0.0, tmp5)
    tmp9 = tl.where(tmp4, tmp7, 0.0)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 128, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr1 + (64*x2 + 4194304*y1 + ((-64) + y0)), tmp10 & ymask & xmask, other=0.0, eviction_policy='evict_first')
    tmp14 = tl.load(in_ptr2 + (tl.broadcast_to((-64) + y0, [YBLOCK, XBLOCK])), tmp10 & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp17 = tl.maximum(0.0, tmp15)
    tmp19 = tl.where(tmp10, tmp17, 0.0)
    tmp20 = tl.where(tmp4, tmp9, tmp19)
    tl.store(out_ptr0 + (x2 + 65536*y3), tmp20, ymask & xmask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 16384
    xnumel = 65536
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_cat_convolution_relu_5[grid](arg_0, arg_1, arg_2, arg_3, 16384, 65536)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((8388608, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 256, 256), (4194304, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 128, 256, 256), (8388608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
