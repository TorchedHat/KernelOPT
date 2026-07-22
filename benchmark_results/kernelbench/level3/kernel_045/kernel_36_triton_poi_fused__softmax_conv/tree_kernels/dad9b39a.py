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
def triton_poi_fused__softmax_convolution_exp_sub_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flatten 2D (ynumel=8192, xnumel=4) into 1D numel=32768
    # Use XBLOCK as the flat BLOCK size, YBLOCK=1 (unused dimension)
    numel = 32768
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    tile_start = pid * XBLOCK
    while tile_start < numel:
        offsets = tile_start + tl.arange(0, XBLOCK)
        mask = offsets < numel
        x2 = offsets % 4
        y3 = offsets // 4
        y0 = y3 % 64
        y1 = y3 // 64
        tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), mask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 64*x2 + 256*y1), tmp0, mask)
        tile_start += num_progs * XBLOCK

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 8192
    xnumel = 4
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_convolution_exp_sub_36[grid](arg_0, arg_1, 8192, 4)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cuda:0', dtype=torch.float32)
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
