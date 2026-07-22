import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'YBLOCK': 64},  num_warps=2),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=8),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_22(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # 1D grid: pid(0) covers Y only; X dimension collapsed into static inner loop
    yoffset = tl.program_id(0) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)  # 1D shape [YBLOCK]
    ymask = yindex < ynumel
    y0 = yindex % 128
    y1 = yindex // 128
    # Inner loop over all 9 X elements (compile-time constant, unrolled)
    for x2 in tl.static_range(9):
        tmp0 = tl.load(in_ptr0 + (x2 + 9 * yindex), mask=ymask, other=0.0)
        tl.store(out_ptr0 + (y0 + 128 * x2 + 1152 * y1), tmp0, mask=ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 24576
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_22[grid](arg_0, arg_1, 24576, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((192, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
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
