import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_addmm_clone_split_transpose_view_4(
        in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    tile_id = pid
    num_sms = tl.num_programs(0)
    xmask = tl.full([XBLOCK], True, tl.int1)
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        x0 = xindex % 96
        x2 = (xindex // 49152) % 8
        x1 = (xindex // 96) % 512
        x3 = xindex // 393216
        x4 = xindex
        tmp0 = tl.load(in_ptr0 + (1536 + x0 + 96 * x2), mask=xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (1536 + x0 + 96 * x2 + 2304 * x1 + 1179648 * x3), mask=xmask)
        tl.store(out_ptr0 + x4, tmp0 + tmp1, mask=xmask)
        tile_id = tile_id + num_sms

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 50331648
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_addmm_clone_split_transpose_view_4[grid](arg_0, arg_1, arg_2, 50331648, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((2304,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((65536, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 8, 512, 96), (393216, 49152, 96, 1), device='cuda:0', dtype=torch.float32)
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
