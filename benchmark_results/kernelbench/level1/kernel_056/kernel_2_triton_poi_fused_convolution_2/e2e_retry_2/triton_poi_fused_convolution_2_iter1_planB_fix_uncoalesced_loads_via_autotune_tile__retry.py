import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 96,  'YBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # xnumel = 96  (channel dimension, stride-1 in NHWC input)
    # ynumel = 774400  (batch * spatial = 256 * 3025)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK] — channel axis
    xmask = xindex < xnumel  # xnumel = 96

    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1] — batch*spatial axis
    ymask = yindex < ynumel  # ynumel = 774400

    c  = xindex           # channel index 0..95, shape [1, XBLOCK]
    hw = yindex % 3025    # spatial index 0..3024, shape [YBLOCK, 1]
    n  = yindex // 3025   # batch index 0..255, shape [YBLOCK, 1]

    # COALESCED load from NHWC (strides: 290400, 1, 5280, 96):
    # element[n, c, h, w] = n*290400 + c*1 + h*5280 + w*96 = n*290400 + c + hw*96
    # Consecutive threads differ by 1 in c (stride-1) → coalesced
    tmp0 = tl.load(in_ptr0 + (c + 96 * hw + 290400 * n), xmask & ymask, eviction_policy='evict_last')

    # Load bias: shape [1, XBLOCK], broadcast over YBLOCK rows
    tmp1 = tl.load(in_ptr1 + c, xmask, eviction_policy='evict_last')

    tmp2 = tmp0 + tmp1

    # Store to NCHW (strides: 290400, 3025, 55, 1):
    # element[n, c, h, w] = n*290400 + c*3025 + h*55 + w = n*290400 + c*3025 + hw
    tl.store(out_ptr0 + (hw + 3025 * c + 290400 * n), tmp2, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
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
