import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32},  num_warps=4),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64},  num_warps=8),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    pid       = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_tiles_x = tl.cdiv(xnumel, XBLOCK)
    num_tiles_y = tl.cdiv(ynumel, YBLOCK)
    num_tiles   = num_tiles_x * num_tiles_y
    tile_id     = pid
    while tile_id < num_tiles:
        y_tile  = tile_id // num_tiles_x
        x_tile  = tile_id %  num_tiles_x
        yoffset = y_tile * YBLOCK
        xoffset = x_tile * XBLOCK
        yindex  = yoffset + tl.arange(0, YBLOCK)[:, None]
        xindex  = xoffset + tl.arange(0, XBLOCK)[None, :]
        ymask   = yindex < ynumel
        xmask   = xindex < xnumel
        x2 = xindex
        y3 = yindex
        y0 = yindex % 128
        y1 = yindex // 128
        tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=ymask & xmask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, mask=ymask & xmask)
        tile_id += grid_size

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7[grid](arg_0, arg_1, 4096, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
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
