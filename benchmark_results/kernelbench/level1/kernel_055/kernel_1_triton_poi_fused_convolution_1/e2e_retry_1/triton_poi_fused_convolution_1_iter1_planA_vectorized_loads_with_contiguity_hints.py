import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    xoffset = tl.program_id(0) * XBLOCK  # tiles y0 (contiguous dim, 0..127)
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK]
    yoffset = tl.program_id(1) * YBLOCK  # tiles x2 (0..31499)
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    ymask = yindex < xnumel  # guard on x2 < 31500
    y0 = tl.max_contiguous(tl.multiple_of(xindex % 128, 128), 128)  # contiguous in in_ptr0 (stride=1)
    y1 = tl.program_id(2)    # batch index (0..31) via grid z-dim
    x2 = yindex              # spatial index (0..31499)
    y3 = y0 + 128 * y1       # output y-index for out_ptr0
    load_idx = y0 + 128*x2 + 4032000*y1
    tmp0 = tl.load(in_ptr0 + load_idx, ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
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
