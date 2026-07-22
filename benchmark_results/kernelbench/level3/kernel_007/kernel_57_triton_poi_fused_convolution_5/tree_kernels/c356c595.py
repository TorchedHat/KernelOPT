import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 128, 'XBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 256, 'XBLOCK': 1}, num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 512, 'XBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 1}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_57(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # X dimension: xnumel=9, use fixed tile of 16 (next power-of-2 >= 9), guarded by xmask
    xindex = tl.arange(0, 16)[None, :]
    xmask = xindex < xnumel
    x2 = xindex

    # Compute total number of y-tiles
    num_tiles_y = (ynumel + YBLOCK - 1) // YBLOCK

    # Persistent loop: each SM strides over y-tiles
    sm_id = tl.program_id(0)
    num_sms = tl.num_programs(0)
    tile_id = sm_id
    while tile_id < num_tiles_y:
        yoffset = tile_id * YBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        ymask = yindex < ynumel
        y3 = yindex
        y0 = yindex % 144
        y1 = yindex // 144
        tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask & ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 144 * x2 + 1296 * y1), tmp0, xmask & ymask)
        tile_id += num_sms

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 41472
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_57[grid](arg_0, arg_1, 41472, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((288, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((288, 144, 3, 3), (1296, 1, 432, 144), device='cuda:0', dtype=torch.float32)
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
