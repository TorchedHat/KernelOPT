import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 128, 'XBLOCK': 1},  num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 256, 'XBLOCK': 1},  num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 512, 'XBLOCK': 1},  num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 4},  num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 256, 'XBLOCK': 4},  num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 256, 'XBLOCK': 8},  num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 256, 'XBLOCK': 16}, num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 512, 'XBLOCK': 16}, num_warps=8, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 9
    num_y_tiles = tl.cdiv(ynumel, YBLOCK)
    num_x_tiles = tl.cdiv(xnumel, XBLOCK)
    num_tiles = num_y_tiles * num_x_tiles
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tile_id = pid
    while tile_id < num_tiles:
        y_tile = tile_id // num_x_tiles
        x_tile = tile_id % num_x_tiles
        yoffset = y_tile * YBLOCK
        xoffset = x_tile * XBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        ymask = yindex < ynumel
        xmask = (xindex < xnumel) & ymask
        x2 = xindex
        y3 = yindex
        y0 = (yindex % 64)
        y1 = yindex // 64
        tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 8192
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_5[grid](arg_0, arg_1, 8192, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
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
