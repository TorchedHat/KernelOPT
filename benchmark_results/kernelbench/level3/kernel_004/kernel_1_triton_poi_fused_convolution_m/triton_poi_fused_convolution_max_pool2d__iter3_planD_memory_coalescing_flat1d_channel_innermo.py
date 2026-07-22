import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=4, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D kernel: treat all output elements as a single 1D range
    # numel = 4096 * 6 * 14 * 14 = 4816896
    # Use program_id(0) only; YBLOCK is constexpr=1 (unused for tiling)
    numel = 4816896
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offsets < numel
    # Decompose flat index channel-innermost: (n, h_out, w_out, c)
    # Output strides: (1176, 84, 6, 1) for (N, H, W, C)
    # c varies fastest -> consecutive threads write consecutive output addresses
    c     = offsets % 6
    tmp   = offsets // 6
    w_out = tmp % 14
    tmp2  = tmp // 14
    h_out = tmp2 % 14
    n     = tmp2 // 14
    # Input base: NCHW strides (4704, 784, 28, 1)
    h_in = h_out * 2
    w_in = w_out * 2
    base = n * 4704 + c * 784 + h_in * 28 + w_in
    # Load 2x2 pool window
    v0 = tl.load(in_ptr0 + base,      mask=mask)  # (h_in,   w_in  )
    v1 = tl.load(in_ptr0 + base + 1,  mask=mask)  # (h_in,   w_in+1)
    v2 = tl.load(in_ptr0 + base + 28, mask=mask)  # (h_in+1, w_in  )
    v3 = tl.load(in_ptr0 + base + 29, mask=mask)  # (h_in+1, w_in+1)
    result = tl.maximum(tl.maximum(v0, v1), tl.maximum(v2, v3))
    # Output address: NHWC strides (1176, 84, 6, 1)
    out_offset = n * 1176 + h_out * 84 + w_out * 6 + c
    tl.store(out_ptr0 + out_offset, result, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 24576
    xnumel = 196
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_max_pool2d_with_indices_relu_1[grid](arg_0, arg_1, 24576, 196)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((4096, 6, 28, 28), (4704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4096, 6, 14, 14), (1176, 1, 84, 6), device='cuda:0', dtype=torch.float32)
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
