import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 64,  'XBLOCK': 1},  num_warps=4),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 1},  num_warps=4),
        triton.Config({'YBLOCK': 256, 'XBLOCK': 1},  num_warps=4),
        triton.Config({'YBLOCK': 512, 'XBLOCK': 1},  num_warps=8),
        triton.Config({'YBLOCK': 1024,'XBLOCK': 1},  num_warps=8),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 16}, num_warps=4),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4),
        triton.Config({'YBLOCK': 256, 'XBLOCK': 16}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_relu_6(in_ptr0, out_ptr0, ynumel, xnumel,
                                         YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Standard 2D tiled — NO persistent loop
    yoffset = tl.program_id(1) * YBLOCK
    xoffset = tl.program_id(0) * XBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # [YBLOCK, 1]
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # [1, XBLOCK]
    ymask = yindex < ynumel
    xmask = xindex < xnumel
    mask = ymask & xmask
    # Load coalesced from input (sequential linear layout: y3*9 + x2)
    tmp0 = tl.load(in_ptr0 + yindex * xnumel + xindex, mask=mask, other=0.0)
    # Decompose y index for output address
    y0 = yindex % 64   # [YBLOCK, 1] — stride-1 in output
    y1 = yindex // 64  # [YBLOCK, 1]
    # With XBLOCK=1: xindex is scalar per CTA, all YBLOCK threads write
    # to y0 + 64*x2_fixed + 576*y1. For yindex=0..63: y0=0..63, y1=0
    # -> addresses 0,1,2,...,63 = stride-1 = fully coalesced!
    tl.store(out_ptr0 + (y0 + 64 * xindex + 576 * y1), tmp0, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 12288
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_relu_6[grid](arg_0, arg_1, 12288, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((192, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
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
