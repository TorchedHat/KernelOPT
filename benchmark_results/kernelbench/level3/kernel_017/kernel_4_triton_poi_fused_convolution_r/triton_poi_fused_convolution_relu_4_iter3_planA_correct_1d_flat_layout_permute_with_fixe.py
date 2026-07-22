import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'YBLOCK': 32},  num_warps=1),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 64},  num_warps=2),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # 1D flat kernel using yindex as the flat index over all 3456 elements
    yoffset = tl.program_id(0) * YBLOCK
    yindex  = yoffset + tl.arange(0, YBLOCK)  # shape [YBLOCK], flat 1D
    ymask   = yindex < ynumel
    # Decompose flat index: in_ptr0 stride=(54,9,3,1) -> C-contiguous
    # flat index i = 54*y1 + 9*y0 + 3*h + w = 9*tmp + x2 = yindex
    x2  = yindex % 9          # spatial hw index 0..8
    tmp = yindex // 9          # 0..383
    y0  = tmp % 6              # channel index 0..5
    y1  = tmp // 6             # batch index 0..63
    # Load: contiguous (input is C-contiguous with stride (54,9,3,1))
    tmp0 = tl.load(in_ptr0 + yindex, mask=ymask, other=0.0)
    # Store: out_ptr0 stride=(54,1,18,6)
    # x2=3*h+w -> 18*(x2//3)+6*(x2%3)=6*x2, so store offset = y0 + 6*x2 + 54*y1
    tl.store(out_ptr0 + (y0 + 6 * x2 + 54 * y1), tmp0, mask=ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 384
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_relu_4[grid](arg_0, arg_1, 384, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((64, 6, 3, 3), (54, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 6, 3, 3), (54, 1, 18, 6), device='cuda:0', dtype=torch.float32)
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
