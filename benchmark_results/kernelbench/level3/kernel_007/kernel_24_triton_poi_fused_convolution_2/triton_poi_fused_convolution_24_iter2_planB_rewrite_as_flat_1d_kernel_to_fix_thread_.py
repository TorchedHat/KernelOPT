import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'YBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 1024}, num_warps=4),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 1024}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flat 1D kernel: treat all 76800 elements as a single dimension
    # numel = ynumel * xnumel = 3072 * 25 = 76800
    pid = tl.program_id(0)
    offsets = pid * YBLOCK + tl.arange(0, YBLOCK)
    numel = 76800
    mask = offsets < numel
    # Flat index decomposition: flat = x2 + 25 * y3
    x2 = offsets % 25          # xindex in [0, 24]
    y3 = offsets // 25         # yindex in [0, 3071]
    y0 = y3 % 32              # y0 = yindex % 32
    y1 = y3 // 32             # y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 25 * y3), mask)
    tl.store(out_ptr0 + (y0 + 32 * x2 + 800 * y1), tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 3072
    xnumel = 25
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_24[grid](arg_0, arg_1, 3072, 25)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((96, 32, 5, 5), (800, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96, 32, 5, 5), (800, 1, 160, 32), device='cuda:0', dtype=torch.float32)
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
