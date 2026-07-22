import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 1024, 'XBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 512,  'XBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 256,  'XBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 128,  'XBLOCK': 1}, num_warps=2, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D kernel: treat all 1,638,400 output elements as a single dimension
    numel = 1638400
    pid = tl.program_id(0)
    offsets = pid * YBLOCK + tl.arange(0, YBLOCK)
    mask = offsets < numel

    # Decompose flat index into (n, c, oh, ow)
    ow = offsets % 5
    tmp = offsets // 5
    oh = tmp % 5
    tmp = tmp // 5
    c = tmp % 16
    n = tmp // 16

    # Input base offset using non-contiguous strides: (1600, 1, 160, 16)
    base = n * 1600 + c * 1
    ih0 = 2 * oh
    iw0 = 2 * ow

    # Load 4 elements of the 2x2 max-pool window
    v00 = tl.load(in_ptr0 + base + ih0 * 160 + iw0 * 16,               mask=mask)
    v01 = tl.load(in_ptr0 + base + ih0 * 160 + (iw0 + 1) * 16,         mask=mask)
    v10 = tl.load(in_ptr0 + base + (ih0 + 1) * 160 + iw0 * 16,         mask=mask)
    v11 = tl.load(in_ptr0 + base + (ih0 + 1) * 160 + (iw0 + 1) * 16,   mask=mask)

    # Max-pool reduction
    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    # Output offset using contiguous strides: (400, 25, 5, 1)
    out_idx = n * 400 + c * 25 + oh * 5 + ow
    tl.store(out_ptr0 + out_idx, result, mask=mask)

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
