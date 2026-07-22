import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8),
        triton.Config({'XBLOCK': 2,  'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 2,  'YBLOCK': 256}, num_warps=8),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32},  num_warps=4),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64},  num_warps=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 25
    pid = tl.program_id(0)
    num_x_tiles = tl.cdiv(xnumel, XBLOCK)
    x_tile_id = pid % num_x_tiles
    y_tile_id = pid // num_x_tiles
    xoffset = x_tile_id * XBLOCK
    yoffset = y_tile_id * YBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    xmask = xindex < xnumel
    ymask = yindex < ynumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 400*y1), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 768
    xnumel = 25
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_36[grid](arg_0, arg_1, 768, 25)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((48, 16, 5, 5), (400, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((48, 16, 5, 5), (400, 1, 80, 16), device='cuda:0', dtype=torch.float32)
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
