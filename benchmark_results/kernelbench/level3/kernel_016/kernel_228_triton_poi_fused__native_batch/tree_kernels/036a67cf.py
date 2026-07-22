import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Persistent tile scheduling: each CTA processes multiple tiles in a loop
    # XBLOCK=1, YBLOCK=128 fixed; tiles_per_row=cdiv(9,1)=9, total_tiles=456*9=4104
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tiles_per_row = tl.cdiv(xnumel, XBLOCK)
    tiles_per_col = tl.cdiv(ynumel, YBLOCK)
    total_tiles = tiles_per_col * tiles_per_row
    tile_id = pid
    while tile_id < total_tiles:
        y_tile = tile_id // tiles_per_row
        x_tile = tile_id % tiles_per_row
        yoffset = y_tile * YBLOCK
        xoffset = x_tile * XBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        ymask = yindex < ynumel
        xmask = xindex < xnumel
        y0 = yindex % 1824
        y1 = yindex // 1824
        tmp0 = tl.load(in_ptr0 + (xindex + 9*yindex), xmask & ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 1824*xindex + 16416*y1), tmp0, xmask & ymask)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 58368
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_228[grid](arg_0, arg_1, 58368, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 1824, 3, 3), (16416, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 1824, 3, 3), (16416, 1, 5472, 1824), device='cuda:0', dtype=torch.float32)
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
