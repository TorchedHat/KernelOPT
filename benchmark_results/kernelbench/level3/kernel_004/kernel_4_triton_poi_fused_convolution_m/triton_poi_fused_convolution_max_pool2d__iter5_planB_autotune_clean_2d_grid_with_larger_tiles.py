import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK':  64, 'XBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK':  32, 'XBLOCK': 16}, num_warps=2, num_stages=3),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=8, num_stages=2),
        triton.Config({'YBLOCK':  64, 'XBLOCK': 16}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Y axis: flattened (N, OH, OW) — 4096*5*5=102400 elements
    # X axis: C — 16 channels
    yoffset = tl.program_id(1) * YBLOCK
    yindex  = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask   = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask   = xindex < xnumel
    mask    = ymask & xmask
    # Decode Y -> (n, oh, ow) using output spatial 5x5=25
    n  = yindex // 25
    rem = yindex % 25
    oh = rem // 5
    ow = rem % 5
    c  = xindex
    # Input strides (1600, 1, 160, 16): base = n*1600 + c + oh*320 + ow*32
    base = n * 1600 + c + oh * 320 + ow * 32
    v00 = tl.load(in_ptr0 + base,       mask=mask)
    v01 = tl.load(in_ptr0 + base + 16,  mask=mask)
    v10 = tl.load(in_ptr0 + base + 160, mask=mask)
    v11 = tl.load(in_ptr0 + base + 176, mask=mask)
    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
    # Output strides (400, 25, 5, 1)
    tl.store(out_ptr0 + n*400 + c*25 + oh*5 + ow, result, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 102400
    xnumel = 16
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_convolution_max_pool2d_with_indices_relu_4[grid](arg_0, arg_1, 102400, 16)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((4096, 16, 10, 10), (1600, 1, 160, 16), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4096, 16, 5, 5), (400, 25, 5, 1), device='cuda:0', dtype=torch.float32)
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
