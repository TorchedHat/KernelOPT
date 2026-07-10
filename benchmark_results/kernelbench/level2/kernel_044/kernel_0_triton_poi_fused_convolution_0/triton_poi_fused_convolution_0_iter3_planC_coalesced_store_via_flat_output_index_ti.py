import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # numel = ynumel * xnumel = 30 * 50176 = 1505280
    # We tile over the flat output index for stride-1 coalesced stores
    numel = ynumel * xnumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xmask = xindex < numel

    # Decode flat OUTPUT index -> (n, c, h, w)
    # Output layout: stride=(150528, 1, 672, 3), shape=(10, 3, 224, 224)
    # out_addr = n*150528 + c*1 + h*672 + w*3 = xindex  (stride-1 in c)
    # Inversion: xindex = n*150528 + c + h*672 + w*3
    #   rem = xindex % 150528 = c + h*672 + w*3  (h*672+w*3 = 3*(h*224+w) always divisible by 3)
    #   c = rem % 3,  hw = rem // 3 = h*224 + w,  h = hw // 224,  w = hw % 224
    n   = xindex // 150528
    rem = xindex % 150528
    c   = rem % 3
    hw  = rem // 3
    h   = hw // 224
    w   = hw % 224

    # Input address: stride=(150528, 50176, 224, 1)
    in_addr = n * 150528 + c * 50176 + h * 224 + w

    tmp0 = tl.load(in_ptr0 + in_addr, xmask)
    tl.store(out_ptr0 + xindex, tmp0, xmask)  # stride-1: fully coalesced!

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
