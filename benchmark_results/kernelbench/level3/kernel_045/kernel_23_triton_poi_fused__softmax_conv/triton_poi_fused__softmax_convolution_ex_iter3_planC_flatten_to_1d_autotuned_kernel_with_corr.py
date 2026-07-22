import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4,  num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D decomposition over all numel = ynumel * xnumel = 9,437,184 elements
    numel = ynumel * xnumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    mask = xindex < numel
    # Decompose flat index into (y3, x2) — x2 varies fastest (matches original inner loop)
    x2 = xindex % 9          # filter position 0..8
    y3 = xindex // 9         # channel pair index 0..1048575
    y0 = y3 % 1024           # output channel 0..1023
    y1 = y3 // 1024          # input channel 0..1023
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 1024 * x2 + 9216 * y1), tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 1048576
    xnumel = 9
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__softmax_convolution_exp_sub_23[grid](arg_0, arg_1, 1048576, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1024, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 1024, 3, 3), (9216, 1, 3072, 1024), device='cuda:0', dtype=torch.float32)
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
