import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_clone_unfold_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)  # = NUM_SMS at runtime
    tile_id = pid  # each SM starts at its own tile
    while tile_id * XBLOCK < xnumel:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = (xindex % 16)
        x1 = ((xindex // 16) % 16)
        x2 = ((xindex // 256) % 14)
        x3 = xindex // 3584
        x4 = xindex
        tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 224*x1 + 3584*x3), xmask)
        tl.store(out_ptr0 + (x4), tmp0, xmask)
        tile_id += num_programs  # stride by grid size

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 301056
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_unfold_0[grid](arg_0, arg_1, 301056)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((2, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2, 3, 14, 14, 16, 16), (150528, 50176, 3584, 256, 16, 1), device='cuda:0', dtype=torch.float32)
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
