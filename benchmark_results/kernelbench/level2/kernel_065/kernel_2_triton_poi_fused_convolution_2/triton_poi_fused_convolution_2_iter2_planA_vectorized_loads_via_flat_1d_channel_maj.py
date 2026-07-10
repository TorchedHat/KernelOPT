import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Axes reassigned:
    #   x-axis (XBLOCK=96) -> channel dim C=96, stride=1 in in_ptr0 -> COALESCED
    #   y-axis (YBLOCK)    -> flattened (batch * spatial) = 256*3025 = 774400
    C = 96
    NHW = 774400  # 256 * 3025

    xoffset = tl.program_id(0) * XBLOCK
    # Apply vectorization hints on 1D arange before reshape
    xarange = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xindex = xarange[None, :]   # shape [1, XBLOCK] — channel index

    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1] — batch*spatial index
    ymask = yindex < NHW        # guard (batch*spatial) index < 774400

    # xmask guards channel_idx < 96 AND batch*spatial < 774400
    xmask = (xindex < C) & ymask

    # Decompose yindex into spatial (x2) and batch (y1)
    x2 = yindex % 3025   # spatial index
    y1 = yindex // 3025  # batch index

    # chan_idx = xindex (channel, stride=1 in in_ptr0 -> COALESCED)
    chan_idx = xindex

    # Load: address = chan_idx + 96*x2 + 290400*y1
    # Identical to original y0 + 96*x2 + 290400*y1
    # chan_idx varies by 1 per thread -> COALESCED 128-bit loads
    tmp0 = tl.load(in_ptr0 + (chan_idx + 96*x2 + 290400*y1),
                   mask=xmask, eviction_policy='evict_last')
    # Bias: per-channel, broadcast over YBLOCK rows; mask by channel guard
    tmp1 = tl.load(in_ptr1 + chan_idx, mask=xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    # Store: address = x2 + 3025*(chan_idx + 96*y1)
    # Identical to original x2 + 3025*y3 where y3 = chan_idx + 96*y1
    tl.store(out_ptr0 + (x2 + 3025*(chan_idx + 96*y1)),
             tmp2, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025, YBLOCK=8, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((256, 96, 55, 55), (290400, 1, 5280, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256, 96, 55, 55), (290400, 3025, 55, 1), device='cuda:0', dtype=torch.float32)
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
