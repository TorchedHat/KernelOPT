import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 96,  'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 96,  'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 96,  'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # X axis covers N*C = 24576 (channel dim, stride-1 in in_ptr0 → coalesced)
    # Y axis covers H*W  = 3025  (spatial dim)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]   # shape (1, XBLOCK), N*C index
    xmask = xindex < 24576                              # N*C boundary mask

    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]   # shape (YBLOCK, 1), spatial index
    ymask = yindex < 3025                               # spatial boundary mask

    # Decompose N*C index into channel (y0) and batch (y1)
    y0 = xindex % 96        # channel index (0..95), stride-1 in in_ptr0
    y1 = xindex // 96       # batch index (0..255)
    y3 = xindex             # N*C combined index for output

    # x2 is the spatial index (H*W position)
    x2 = yindex             # shape (YBLOCK, 1)

    # Load from NHWC input: address = channel + 96*spatial + 290400*batch
    # y0 shape (1,XBLOCK), x2 shape (YBLOCK,1), y1 shape (1,XBLOCK)
    # → broadcasts to (YBLOCK, XBLOCK); consecutive X-threads access consecutive channels → COALESCED
    tmp0 = tl.load(in_ptr0 + (y0 + 96 * x2 + 290400 * y1), xmask & ymask, eviction_policy='evict_last')

    # Load bias: indexed by channel only (y0 shape (1,XBLOCK))
    tmp1 = tl.load(in_ptr1 + y0, xmask, eviction_policy='evict_last')

    tmp2 = tmp0 + tmp1

    # Store to NCHW output: address = spatial + 3025*(N*C index)
    # x2 shape (YBLOCK,1), y3 shape (1,XBLOCK) → broadcasts to (YBLOCK, XBLOCK)
    tl.store(out_ptr0 + (x2 + 3025 * y3), tmp2, xmask & ymask)

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
