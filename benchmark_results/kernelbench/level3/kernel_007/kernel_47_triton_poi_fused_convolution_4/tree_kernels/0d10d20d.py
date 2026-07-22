import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,   'YBLOCK': 1}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_47(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 1536
    xnumel = 25
    numel = ynumel * xnumel  # 38400
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    # Persistent loop: each CTA strides through the flat index space
    tile_start = pid * XBLOCK
    tile_stride = num_programs * XBLOCK
    for flat_start in range(tile_start, numel, tile_stride):
        flat_idx = flat_start + tl.arange(0, XBLOCK)
        mask = flat_idx < numel
        # Decompose flat index into original 2D indices
        x2 = flat_idx % 25       # original xindex
        y3 = flat_idx // 25      # original yindex
        y0 = y3 % 24
        y1 = y3 // 24
        tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), mask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 24*x2 + 600*y1), tmp0, mask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 1536
    xnumel = 25
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_47[grid](arg_0, arg_1, 1536, 25)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((64, 24, 5, 5), (600, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 24, 5, 5), (600, 1, 120, 24), device='cuda:0', dtype=torch.float32)
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
