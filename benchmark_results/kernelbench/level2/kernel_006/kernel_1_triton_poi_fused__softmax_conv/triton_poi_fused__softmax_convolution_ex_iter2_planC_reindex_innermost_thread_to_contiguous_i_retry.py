import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 512}, num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # xnumel = 128 * 16 * 14 * 30 * 30 = 25804800 (full input spatial volume)
    xnumel = 25804800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    # Innermost = input W (stride-1 in in_ptr0) -- COALESCED
    x0 = xindex % 30           # input col  (stride-1 in memory)
    x1 = (xindex // 30) % 30   # input row
    x2 = (xindex // 900) % 14  # input depth
    x3 = xindex // 12600       # batch * channel
    # Single coalesced load
    val = tl.load(in_ptr0 + (x0 + 30*x1 + 900*x2 + 12608*x3), xmask)
    # Map to output position (floor-divide by pool stride=2)
    out_x0 = x0 // 2   # output W (0..14)
    out_x1 = x1 // 2   # output H (0..14)
    out_x2 = x2 // 2   # output D (0..6)
    # Output strides: (25600, 1600, 225, 15, 1) for (batch, ch, D, H, W)
    # x3 = batch*channel index, output stride per batch*channel = 1600
    out_idx = out_x0 + 15*out_x1 + 225*out_x2 + 1600*x3
    # Atomic max accumulates the 8 input contributions per output cell
    tl.atomic_max(out_ptr0 + out_idx, val, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 3225600
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1[grid](arg_0, arg_1, 3225600)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 16, 14, 30, 30), (201728, 12608, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 7, 15, 15), (25600, 1600, 225, 15, 1), device='cuda:0', dtype=torch.float32)
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
