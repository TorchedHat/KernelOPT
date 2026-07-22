import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1024,'YBLOCK': 1}, num_warps=8, num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_59(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D indexing over numel = ynumel * xnumel = 51200 elements
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    numel = ynumel * xnumel
    mask = xindex < numel
    # Recover 2D indices: flat index = y3*25 + x2
    x2 = xindex % xnumel
    y3 = xindex // xnumel
    y0 = y3 % 32
    y1 = y3 // 32
    # Load: in_ptr0 stride is (800,25,5,1); x2+25*y3 = xindex (row-major)
    tmp0 = tl.load(in_ptr0 + xindex, mask=mask, eviction_policy='evict_last')
    # Store: transposed layout - streaming write, use evict_first
    tl.store(out_ptr0 + (y0 + 32 * x2 + 800 * y1), tmp0, mask=mask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 2048
    xnumel = 25
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_59[grid](arg_0, arg_1, 2048, 25)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((64, 32, 5, 5), (800, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 32, 5, 5), (800, 1, 160, 32), device='cuda:0', dtype=torch.float32)
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
