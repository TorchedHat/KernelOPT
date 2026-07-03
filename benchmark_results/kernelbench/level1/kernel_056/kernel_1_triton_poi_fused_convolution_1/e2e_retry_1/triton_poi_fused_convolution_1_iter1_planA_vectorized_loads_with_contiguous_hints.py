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
    # XBLOCK now covers the 128-channel+batch flat dim (stride-1 in input) — fully coalesced
    # YBLOCK now tiles the 31500-spatial dim
    xoffset = tl.program_id(0) * XBLOCK
    # Apply vectorization hints on 1D arange before reshaping to 2D
    xarange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), 128), 128)
    xindex = xarange[None, :]   # shape [1, XBLOCK]
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]   # shape [YBLOCK, 1]
    ymask = yindex < xnumel  # guard spatial dim (31500 not power-of-2)
    y0 = xindex % 128      # channel index (0..127), stride-1 in input
    y1 = xindex // 128     # batch index (0..31)
    x2 = yindex            # spatial index
    y3 = xindex            # flat output channel index
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), ymask, eviction_policy='evict_last')
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
