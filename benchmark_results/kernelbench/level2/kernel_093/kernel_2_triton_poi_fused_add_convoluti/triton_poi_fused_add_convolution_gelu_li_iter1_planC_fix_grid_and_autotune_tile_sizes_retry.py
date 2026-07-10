import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 2},  num_warps=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 4},  num_warps=4),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 8,   'YBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=8),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 2},  num_warps=8),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 4},  num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_add_convolution_gelu_lift_fresh_minimum_mul_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2163200
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16900)
    y1 = yindex // 16900
    tmp0 = tl.load(in_ptr0 + (x2 + 128*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0.5, tl.float32)
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1, 1], 0.0, tl.float32)
    tmp6 = tl.minimum(tmp4, tmp5)
    tmp7 = tmp6 * tmp3
    tmp8 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = libdevice.erf(tmp9)
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp7 * tmp12
    tmp14 = tl.full([1, 1], 2.0, tl.float32)
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + (y0 + 16900*x2 + 2163200*y1), tmp15, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 2163200
    xnumel = 128
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_add_convolution_gelu_lift_fresh_minimum_mul_2[grid](arg_0, arg_1, arg_2, 2163200, 128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 128, 130, 130), (2163200, 1, 16640, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 128, 130, 130), (2163200, 16900, 130, 1), device='cuda:0', dtype=torch.float32)
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
