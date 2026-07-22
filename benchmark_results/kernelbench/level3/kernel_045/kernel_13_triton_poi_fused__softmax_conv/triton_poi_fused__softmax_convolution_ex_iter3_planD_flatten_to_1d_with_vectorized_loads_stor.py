import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 512,  'XBLOCK': 1}, num_warps=4),
        triton.Config({'YBLOCK': 1024, 'XBLOCK': 1}, num_warps=8),
        triton.Config({'YBLOCK': 2048, 'XBLOCK': 1}, num_warps=8),
        triton.Config({'YBLOCK': 2048, 'XBLOCK': 1}, num_warps=16),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 65536
    xnumel = 9
    numel = 589824
    pid = tl.program_id(0)
    offsets = pid * YBLOCK + tl.arange(0, YBLOCK)
    mask = offsets < numel
    x2 = offsets % 9
    y3 = offsets // 9
    y0 = y3 % 256
    y1 = y3 // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256 * x2 + 2304 * y1), tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 65536
    xnumel = 9
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__softmax_convolution_exp_sub_13[grid](arg_0, arg_1, 65536, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
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
