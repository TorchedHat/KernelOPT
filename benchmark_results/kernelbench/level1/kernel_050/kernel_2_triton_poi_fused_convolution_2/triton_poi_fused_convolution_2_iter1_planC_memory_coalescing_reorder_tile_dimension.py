import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 96      # channel dimension — stride-1 in in_ptr0
    ynumel = 776800  # batch * spatial = 256 * 3025

    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # channel: stride-1 in input
    xmask = xindex < xnumel  # needed since 96 is not power-of-2

    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # batch*spatial
    ymask = yindex < ynumel

    c  = xindex          # channel index, 0..95
    sp = yindex % 3025   # spatial index, 0..3024
    n  = yindex // 3025  # batch index, 0..255

    # Load: in_ptr0 strides (290400, 1, 5280, 96) -> offset = c + 96*sp + 290400*n
    # Consecutive threads vary in c (xindex) -> stride-1 -> COALESCED
    tmp0 = tl.load(in_ptr0 + (c + 96*sp + 290400*n), xmask & ymask)

    # Bias: in_ptr1[c], broadcast over spatial/batch
    tmp1 = tl.load(in_ptr1 + c, xmask, eviction_policy='evict_last')

    tmp2 = tmp0 + tmp1

    # Store: out_ptr0 strides (290400, 3025, 55, 1) -> offset = sp + 3025*c + 290400*n
    tl.store(out_ptr0 + (sp + 3025*c + 290400*n), tmp2, xmask & ymask)

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
