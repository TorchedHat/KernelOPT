import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'YBLOCK': 128},  num_warps=4),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 2048}, num_warps=16),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # numel = ynumel * xnumel = 12288 * 9 = 110592
    # Use 1D flat-index tiling via YBLOCK (XBLOCK=1 in all configs)
    numel = 110592
    xnumel_const = 9
    pid = tl.program_id(0)
    flat_offset = pid * YBLOCK
    flat_index = flat_offset + tl.arange(0, YBLOCK)
    mask = flat_index < numel
    # Recover 2D indices from flat: flat_index = y3 * xnumel + x2
    x2 = flat_index % xnumel_const          # kernel spatial index [0..8]
    y3 = flat_index // xnumel_const         # combined batch*channel index [0..12287]
    y0 = y3 % 96                            # output channel index
    y1 = y3 // 96                           # batch index
    # Load: in_ptr0 stride (864,9,3,1) -> address = x2 + 9*y3
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), mask=mask, eviction_policy='evict_last')
    # Store: out_ptr0 stride (864,1,288,96) -> address = y0 + 96*x2 + 864*y1
    tl.store(out_ptr0 + (y0 + 96 * x2 + 864 * y1), tmp0, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 12288
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
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
