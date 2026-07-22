import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_53(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 32768
    xnumel = 9
    sm_id = tl.program_id(0)  # 0..num_sms-1, 1D grid
    num_sms = tl.num_programs(0)  # number of CTAs launched (132)
    num_y_tiles = tl.cdiv(ynumel, YBLOCK)  # 4096
    xindex = tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK]
    xmask = xindex < xnumel  # guard for indices 9..15
    # Persistent loop: each SM strides through Y-tiles
    for tile_id in range(sm_id, num_y_tiles, num_sms):
        yoffset = tile_id * YBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
        y0 = yindex % 128
        y1 = yindex // 128
        tmp0 = tl.load(in_ptr0 + (xindex + 9*yindex), xmask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 128*xindex + 1152*y1), tmp0, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 32768
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_53[grid](arg_0, arg_1, 32768, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
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
