import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D kernel over numel = ynumel * xnumel = 131072 * 9 = 1179648
    # YBLOCK is always 1; XBLOCK is the effective block size
    numel = 131072 * 9
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offsets < numel
    # Recover 2D indices from flat index: flat = x2 + 9*y3
    x2 = offsets % 9          # kernel spatial index [0..8]
    y3 = offsets // 9         # combined (n, c) index [0..131071]
    y0 = y3 % 512             # channel index [0..511]
    y1 = y3 // 512            # batch index [0..255]
    # Load from contiguous NCHW layout (stride 4608, 9, 3, 1)
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), mask=mask)
    # Store to permuted layout (stride 4608, 1, 1536, 512)
    tl.store(out_ptr0 + (y0 + 512 * x2 + 4608 * y1), tmp0, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 131072
    xnumel = 9
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_convolution_31[grid](arg_0, arg_1, 131072, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
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
