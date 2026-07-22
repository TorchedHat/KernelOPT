import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    num_xtiles = tl.cdiv(xnumel, XBLOCK)   # cdiv(9,16)=1
    num_ytiles = tl.cdiv(ynumel, YBLOCK)   # cdiv(15360,64)=240
    total_tiles = num_xtiles * num_ytiles   # 240
    num_sms = tl.num_programs(0)            # = 132 (grid size)
    sm_id = tl.program_id(0)               # 0..131
    tile_id = sm_id
    while tile_id < total_tiles:
        ytile = tile_id // num_xtiles
        xtile = tile_id % num_xtiles
        yindex = ytile * YBLOCK + tl.arange(0, YBLOCK)[:, None]
        xindex = xtile * XBLOCK + tl.arange(0, XBLOCK)[None, :]
        ymask = yindex < ynumel
        xmask = xindex < xnumel
        mask = xmask & ymask
        y0 = yindex % 480
        y1 = yindex // 480
        tmp0 = tl.load(in_ptr0 + (xindex + 9 * yindex), mask=mask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 480 * xindex + 4320 * y1), tmp0, mask=mask, eviction_policy='evict_first')
        tile_id += num_sms

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 15360
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40[grid](arg_0, arg_1, 15360, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 480, 3, 3), (4320, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 480, 3, 3), (4320, 1, 1440, 480), device='cuda:0', dtype=torch.float32)
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
