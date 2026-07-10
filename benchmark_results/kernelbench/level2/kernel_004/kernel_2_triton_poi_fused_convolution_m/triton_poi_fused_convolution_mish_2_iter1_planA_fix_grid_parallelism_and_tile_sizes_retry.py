import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_mish_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4129024
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64516)
    y1 = yindex // 64516
    tmp0 = tl.load(in_ptr0 + (x2 + 128*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 20.0, tl.float32)
    tmp4 = tmp2 > tmp3
    tmp5 = tl.exp(tmp2)
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 > tmp3
    tmp11 = tl.exp(tmp9)
    tmp12 = libdevice.log1p(tmp11)
    tmp13 = tl.where(tmp10, tmp9, tmp12)
    tmp14 = libdevice.tanh(tmp13)
    tmp15 = tmp9 * tmp14
    tl.store(out_ptr0 + (y0 + 64516*x2 + 8258048*y1), tmp15, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 4129024
    xnumel = 128
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_convolution_mish_2[grid](arg_0, arg_1, arg_2, 4129024, 128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((64, 128, 254, 254), (8258048, 1, 32512, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64, 128, 254, 254), (8258048, 64516, 254, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
