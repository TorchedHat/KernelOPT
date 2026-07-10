import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=8, num_stages=4),
    triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=8, num_stages=4),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    xoffset = tl.program_id(0) * YBLOCK
    xindex = xoffset + tl.arange(0, YBLOCK)[:, None]   # x2: spatial (outer)
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * XBLOCK
    yindex = yoffset + tl.arange(0, XBLOCK)[None, :]   # y0+y1: channel+batch (warp-parallel)
    # Apply contiguity hints to yindex before decomposing — yindex is contiguous with stride 1
    yindex_cont = tl.max_contiguous(tl.multiple_of(yindex, XBLOCK), XBLOCK)
    y0 = yindex_cont % 128   # channel dim stride-1 in input -> warp-contiguous
    y1 = yindex_cont // 128  # batch dim
    y3 = yindex
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 128, 126, 250), (4032000, 1, 32000, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 128, 126, 250), (4032000, 31500, 250, 1), device='cuda:0', dtype=torch.float32)
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
