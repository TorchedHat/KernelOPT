import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 16,  'XBLOCK': 16}, num_warps=2, num_stages=1),
        triton.Config({'YBLOCK': 32,  'XBLOCK': 16}, num_warps=2, num_stages=1),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'YBLOCK': 256, 'XBLOCK': 16}, num_warps=8, num_stages=1),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 16}, num_warps=2, num_stages=2),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    num_y_tiles = tl.cdiv(ynumel, YBLOCK)
    num_x_tiles = tl.cdiv(xnumel, XBLOCK)
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    total_tiles = num_x_tiles * num_y_tiles
    tile_id = pid
    while tile_id < total_tiles:
        x_tile = tile_id % num_x_tiles
        y_tile = tile_id // num_x_tiles
        xoffset = x_tile * XBLOCK
        yoffset = y_tile * YBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        xmask = xindex < xnumel
        ymask = yindex < ynumel
        y0 = yindex % 224
        y1 = yindex // 224
        tmp0 = tl.load(in_ptr0 + (xindex + 9 * yindex), mask=xmask & ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 224 * xindex + 2016 * y1), tmp0, mask=xmask & ymask)
        tile_id += num_pids

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 7168
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14[grid](arg_0, arg_1, 7168, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 224, 3, 3), (2016, 1, 672, 224), device='cuda:0', dtype=torch.float32)
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
