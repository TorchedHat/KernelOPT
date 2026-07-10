import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_sms = tl.num_programs(0)
    num_tiles = tl.cdiv(xnumel, XBLOCK)  # 8388608 / 1024 = 8192 tiles
    tile_id = pid
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        x0 = xindex % 1024   # index into in_ptr0 (1024 floats, broadcast)
        x1 = xindex // 1024  # index into in_ptr1 (8192 floats, broadcast)
        tmp0 = tl.load(in_ptr0 + x0, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + x1, eviction_policy='evict_last')
        tl.store(out_ptr0 + xindex, tmp0 + tmp1)
        tile_id += num_sms

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 8388608
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_4[grid](arg_0, arg_1, arg_2, 8388608, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((1024, 1), (1, 1024), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 8192, 1, 1), (8192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 8192, 1024, 1), (8388608, 1024, 1, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
