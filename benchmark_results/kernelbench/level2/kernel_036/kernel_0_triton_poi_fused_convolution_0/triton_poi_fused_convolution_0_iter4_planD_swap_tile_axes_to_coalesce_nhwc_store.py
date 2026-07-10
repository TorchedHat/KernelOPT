import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=2),
        triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # numel = ynumel * xnumel = 30 * 50176 = 1,505,280
    # We tile over the flat OUTPUT index for coalesced stores
    # Output strides: (150528, 1, 672, 3) NHWC -> flat index is stride-1
    numel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    # xindex = flat OUTPUT index; consecutive values are stride-1 in output memory -> coalesced store
    xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xmask = xindex < numel
    # Decompose flat output index into (n, c, h, w) using NHWC shape
    # Output layout: [N=10, H=224, W=224, C=3] with strides (150528, 672, 3, 1)
    # flat = n*150528 + h*672 + w*3 + c
    c = xindex % 3
    tmp1 = xindex // 3          # = n*50176 + h*224 + w
    w = tmp1 % 224
    tmp2 = tmp1 // 224          # = n*224 + h
    h = tmp2 % 224
    n = tmp2 // 224
    # Input strides: (150528, 50176, 224, 1) NCHW
    in_idx = n * 150528 + c * 50176 + h * 224 + w
    tmp0 = tl.load(in_ptr0 + in_idx, xmask)
    tl.store(out_ptr0 + xindex, tmp0, xmask)  # stride-1 COALESCED

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
