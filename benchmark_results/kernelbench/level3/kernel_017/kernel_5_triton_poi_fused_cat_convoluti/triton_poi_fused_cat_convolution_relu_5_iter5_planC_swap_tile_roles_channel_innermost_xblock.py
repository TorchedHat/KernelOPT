import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=4, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},   num_warps=4, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},   num_warps=4, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=8, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=8, num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=4, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=8, num_stages=4, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8, num_stages=4, maxnreg=64),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 16},  num_warps=4, num_stages=3, maxnreg=64),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_cat_convolution_relu_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # xnumel = 128 (channels), ynumel = 128*65536 = 8388608 (batch * spatial)
    xnumel = 128
    ynumel = 8388608
    # XBLOCK covers channel dimension (stride-1 in in_ptr0/in_ptr1) -> coalesced
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    # YBLOCK covers batch*spatial dimension
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel

    # c: channel index [0, 128)
    c = xindex
    # s: flat batch*spatial index [0, 128*65536)
    s = yindex
    # batch and spatial decomposition
    batch = s // 65536
    spatial = s % 65536

    # Channel branch: c < 64 -> in_ptr0 (no bias), c >= 64 -> in_ptr1 + in_ptr2 bias
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = c < tmp3   # c in [0, 64)
    tmp10 = c >= tmp3  # c in [64, 128)

    # Load from in_ptr0 (coalesced: XBLOCK threads step by 1 in c)
    # in_ptr0 shape (8388608, 64) strides (64, 1): address = 64*spatial + 4194304*batch + c
    tmp5 = tl.load(in_ptr0 + (64*spatial + 4194304*batch + c), mask=tmp4 & ymask & xmask, other=0.0, eviction_policy='evict_first')
    tmp7 = tl.maximum(0.0, tmp5)
    tmp9 = tl.where(tmp4, tmp7, 0.0)

    # Load from in_ptr1 (coalesced: XBLOCK threads step by 1 in (c-64))
    # in_ptr1 shape (128, 64, 256, 256) strides (4194304, 1, 16384, 64): address = 64*spatial + 4194304*batch + (c-64)
    tmp13 = tl.load(in_ptr1 + (64*spatial + 4194304*batch + (c - 64)), mask=tmp10 & ymask & xmask, other=0.0, eviction_policy='evict_first')
    # Load bias from in_ptr2 (shape (64,) stride (1,)): index = c-64
    tmp14 = tl.load(in_ptr2 + tl.broadcast_to(c - 64, [YBLOCK, XBLOCK]), mask=tmp10 & xmask, other=0.0, eviction_policy='evict_last')
    tmp15 = tmp13 + tmp14
    tmp17 = tl.maximum(0.0, tmp15)
    tmp19 = tl.where(tmp10, tmp17, 0.0)

    tmp20 = tl.where(tmp4, tmp9, tmp19)

    # Store: out_ptr0 shape (128, 128, 256, 256) strides (8388608, 65536, 256, 1)
    # address = spatial + 65536*(batch*128 + c)
    tl.store(out_ptr0 + (spatial + 65536*(batch*128 + c)), tmp20, mask=ymask & xmask)

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
