import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 9
    num_tiles_x = 9   # cdiv(xnumel=9, XBLOCK=1)
    num_tiles_y = 96  # cdiv(ynumel=6144, YBLOCK=64)
    num_tiles = 864   # num_tiles_x * num_tiles_y
    NUM_SMS = 132
    sm_id = tl.program_id(0)
    tile_id = sm_id
    while tile_id < num_tiles:
        x_tile = tile_id % num_tiles_x
        y_tile = tile_id // num_tiles_x
        xoffset = x_tile * XBLOCK
        yoffset = y_tile * YBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        xmask = xindex < xnumel
        ymask = yindex < ynumel
        x2 = xindex
        y3 = yindex
        y0 = (yindex % 192)
        y1 = yindex // 192
        tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 192*x2 + 1728*y1), tmp0, xmask & ymask)
        tile_id += NUM_SMS

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 6144
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12[grid](arg_0, arg_1, 6144, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
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
