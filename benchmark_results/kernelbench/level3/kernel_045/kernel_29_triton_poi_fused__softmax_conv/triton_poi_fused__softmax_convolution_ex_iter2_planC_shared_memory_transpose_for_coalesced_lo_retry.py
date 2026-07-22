import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_29(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 2048

    yoffset = tl.program_id(1) * YBLOCK
    xoffset = tl.program_id(0) * XBLOCK

    # 1D index arrays
    yidx = yoffset + tl.arange(0, YBLOCK)   # [YBLOCK]
    xidx = xoffset + tl.arange(0, XBLOCK)   # [XBLOCK]

    # Decompose y into channel (y0, stride-1 in input) and batch (y1)
    y0 = yidx % 256    # [YBLOCK]
    y1 = yidx // 256   # [YBLOCK]

    ymask = yidx < ynumel   # [YBLOCK]
    xmask = xidx < xnumel   # [XBLOCK]
    mask2d = ymask[:, None] & xmask[None, :]  # [YBLOCK, XBLOCK]

    # LOAD: [YBLOCK, XBLOCK] tile
    # Address: y0[i] + 256*xidx[j] + 524288*y1[i]
    # y0 varies along rows (dim 0) -> stride-1 in in_ptr0 -> COALESCED along y0
    load_addr = (y0[:, None]
                 + 256 * xidx[None, :]
                 + 524288 * y1[:, None])
    tmp0 = tl.load(in_ptr0 + load_addr, mask2d, other=0.0)

    # Bias: indexed by y0 (channel), broadcast over x dim
    tmp1 = tl.load(in_ptr1 + y0, ymask, eviction_policy='evict_last')  # [YBLOCK]
    tmp2 = tmp0 + tmp1[:, None]   # [YBLOCK, XBLOCK]

    # STORE: [YBLOCK, XBLOCK] tile
    # Address: xidx[j] + 2048*y0[i] + 1048576*y1[i]
    # xidx varies along cols (dim 1) -> stride-1 in out_ptr0 -> COALESCED along x
    store_addr = (xidx[None, :]
                  + 2048 * y0[:, None]
                  + 1048576 * y1[:, None])
    tl.store(out_ptr0 + store_addr, tmp2, mask2d)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 2048
    xnumel = 2048
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__softmax_convolution_exp_sub_29[grid](arg_0, arg_1, arg_2, 2048, 2048, YBLOCK=8, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((8, 256, 16, 128), (524288, 1, 32768, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8, 256, 16, 128), (1048576, 2048, 128, 1), device='cuda:0', dtype=torch.float32)
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
