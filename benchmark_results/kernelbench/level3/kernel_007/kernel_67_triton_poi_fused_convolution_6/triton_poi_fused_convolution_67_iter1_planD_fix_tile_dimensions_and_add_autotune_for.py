import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=2),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_67(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flatten to 1D: numel = ynumel * xnumel = 51200 * 9 = 460800
    # Use program_id(0) over the flat space, YBLOCK=1 so program_id(1) is always 0
    numel = ynumel * xnumel
    flat_offset = tl.program_id(0) * XBLOCK
    flat_index = flat_offset + tl.arange(0, XBLOCK)
    xmask = flat_index < numel
    # Recover original 2D indices from flat index
    x2 = flat_index % 9          # kernel spatial index (0..8)
    y3 = flat_index // 9         # combined channel index (0..51199)
    y0 = y3 % 160                # input channel index
    y1 = y3 // 160               # output channel / batch index
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 51200
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_67[grid](arg_0, arg_1, 51200, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((320, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((320, 160, 3, 3), (1440, 1, 480, 160), device='cuda:0', dtype=torch.float32)
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
