import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # XBLOCK=128 covers channel dim (96 elements, needs xmask), YBLOCK covers spatial*batch
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=3),
        # XBLOCK=64 covers channel dim (96 elements, needs 2 blocks), YBLOCK covers spatial*batch
        triton.Config({'XBLOCK': 64,  'YBLOCK': 1},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 1},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 1},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 4},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Axis-swapped: xnumel=96 (channel, stride-1 in NHWC), ynumel=774400 (N*H*W)
    ynumel = 774400
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    # x0: channel index (stride-1 in NHWC input)
    x0 = xindex
    # y_spatial: HW position (0..3024), y_batch: batch index (0..255)
    y_spatial = yindex % 3025
    y_batch = yindex // 3025
    # Load from NHWC input: in_ptr0 strides (290400, 1, 5280, 96)
    # = batch*290400 + channel*1 + spatial*96
    # Consecutive threads vary along x0 (channel, stride-1) -> COALESCED
    tmp0 = tl.load(in_ptr0 + (x0 + 96*y_spatial + 290400*y_batch), mask=xmask & ymask, eviction_policy='evict_first')
    # Load bias indexed by channel (x0), broadcast over YBLOCK
    tmp1 = tl.load(in_ptr1 + x0, mask=xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    # Store to NCHW output: out_ptr0 strides (290400, 3025, 55, 1)
    # = batch*290400 + channel*3025 + spatial*1
    tl.store(out_ptr0 + (y_spatial + 3025*x0 + 290400*y_batch), tmp2, mask=xmask & ymask, eviction_policy='evict_first')

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
