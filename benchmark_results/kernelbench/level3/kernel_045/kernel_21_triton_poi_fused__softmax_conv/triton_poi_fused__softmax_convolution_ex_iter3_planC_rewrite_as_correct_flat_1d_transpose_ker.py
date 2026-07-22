import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 1024, 'XBLOCK': 1}, num_warps=4),
        triton.Config({'YBLOCK': 2048, 'XBLOCK': 1}, num_warps=8),
        triton.Config({'YBLOCK': 512,  'XBLOCK': 1}, num_warps=4),
        triton.Config({'YBLOCK': 4096, 'XBLOCK': 1}, num_warps=8),
        triton.Config({'YBLOCK': 256,  'XBLOCK': 1}, num_warps=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 524288
    xnumel = 9
    N = ynumel * xnumel  # 4718592
    pid = tl.program_id(0)
    offsets = pid * YBLOCK + tl.arange(0, YBLOCK)
    mask = offsets < N
    # Recover 2D indices from flat offset (xnumel=9 is inner dimension)
    x2 = offsets % 9    # column in [0,8]
    y3 = offsets // 9   # row in [0,524287]
    # Sub-indices for output address (IDENTICAL to original formulas)
    y0 = y3 % 512
    y1 = y3 // 512
    # Load: in_ptr0 stride (4608,9,3,1) -> flat index x2 + 9*y3
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), mask=mask)
    # Store: out_ptr0 stride (4608,1,1536,512) -> flat index y0 + 512*x2 + 4608*y1
    tl.store(out_ptr0 + (y0 + 512 * x2 + 4608 * y1), tmp0, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 524288
    xnumel = 9
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_21[grid](arg_0, arg_1, 524288, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
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
