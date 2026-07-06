import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # existing configs (unchanged)
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 64}, num_warps=4,  num_stages=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4,  num_stages=5),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4,  num_stages=5),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8,  num_stages=5),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=8,  num_stages=5),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 64}, num_warps=16, num_stages=5),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=16, num_stages=4),
        # NEW: larger YBLOCK=128 to group more channel-adjacent threads per CTA
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=16, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=16, num_stages=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},   num_warps=8,  num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 3025
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 96)
    y1 = yindex // 96
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), mask=ymask & xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), mask=ymask & xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    # Hint: x2 is 2D [YBLOCK, XBLOCK] shaped — use [1, XBLOCK] to match dims
    x2_hint = tl.max_contiguous(tl.multiple_of(x2, [1, XBLOCK]), [1, XBLOCK])
    tl.store(out_ptr0 + (x2_hint + 3025*y3), tmp2, mask=ymask & xmask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((256, 96, 55, 55), (290400, 1, 5280, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256, 96, 55, 55), (290400, 3025, 55, 1), device='cuda:0', dtype=torch.float32)
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
