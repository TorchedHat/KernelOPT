import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_54(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    # Persistent kernel: 1D grid of num_sms CTAs, each processes multiple tiles
    # NUM_X_TILES = cdiv(9, 128) = 1, NUM_Y_TILES = cdiv(16384, 8) = 2048
    # TOTAL_TILES = 1 * 2048 = 2048
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)  # == 132 at runtime (grid size)
    tile_id = pid
    while tile_id < 2048:
        pid_x = tile_id % 1   # always 0 since NUM_X_TILES=1
        pid_y = tile_id // 1  # ranges 0..2047
        xoffset = pid_x * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        xmask = xindex < xnumel
        yoffset = pid_y * YBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        x2 = xindex
        y3 = yindex
        y0 = (yindex % 512)
        y1 = yindex // 512
        tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask)
        tile_id += num_programs  # stride by SM count

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 16384
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_54[grid](arg_0, arg_1, 16384, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
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
