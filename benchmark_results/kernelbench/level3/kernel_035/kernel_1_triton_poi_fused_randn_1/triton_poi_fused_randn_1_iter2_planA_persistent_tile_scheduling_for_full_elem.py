import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_randn_1(in_ptr0, out_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset, mask=True)
    tile_start = pid * XBLOCK
    tile_stride = grid_size * XBLOCK
    for tile_offset in range(tile_start, xnumel, tile_stride):
        xindex = tile_offset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp2 = tl.randn(tmp0, (x0).to(tl.uint32))
        tl.store(out_ptr0 + (x0), tmp2, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    load_seed_offset = 15360
    xnumel = 1024
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_randn_1[grid](arg_0, arg_1, 15360, 1024, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((2,), (1,), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((6, 10, 256), (2560, 256, 1), device='cuda:0', dtype=torch.float32)
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
