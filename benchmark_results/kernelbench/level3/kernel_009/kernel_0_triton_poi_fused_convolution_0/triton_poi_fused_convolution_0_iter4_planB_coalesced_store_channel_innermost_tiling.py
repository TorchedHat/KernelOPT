import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=1),
    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=1),
    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=8, num_stages=2),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # numel = 10 * 3 * 224 * 224 = 1505280
    # Output strides: (150528, 1, 672, 3) for (N, C, H, W)
    # Input  strides: (150528, 50176, 224, 1) for (N, C, H, W)
    numel = 1505280
    out_linear = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    mask = out_linear < numel
    # Decompose output linear index -> (n, c, h, w)
    # Output layout: out_linear = n*150528 + c*1 + h*672 + w*3
    #              = n*150528 + c + 3*(h*224 + w)
    n   = out_linear // 150528
    rem = out_linear % 150528   # = c + 3*(h*224 + w)
    c   = rem % 3               # channel: output stride=1 (unit stride!)
    hw  = rem // 3              # = h*224 + w
    h   = hw // 224
    w   = hw % 224
    # Input index using NCHW strides (150528, 50176, 224, 1)
    in_idx = n * 150528 + c * 50176 + h * 224 + w
    tmp0 = tl.load(in_ptr0 + in_idx, mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + out_linear, tmp0, mask)  # COALESCED: consecutive addresses

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 30
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
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
