import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64,   'YBLOCK': 1}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 1}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30
    xnumel = 50176
    # 2D grid: hardware computes tile indices, zero runtime division
    xoffset = tl.program_id(0) * XBLOCK
    yoffset = tl.program_id(1) * YBLOCK
    # Build 1D xindex first, then reshape to [1, XBLOCK] for 2D broadcasting
    xindex_1d = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)
    xindex = tl.reshape(xindex_1d, [1, XBLOCK])   # [1, XBLOCK]
    # Build yindex as [YBLOCK, 1]
    yindex = tl.reshape(yoffset + tl.arange(0, YBLOCK), [YBLOCK, 1])  # [YBLOCK, 1]
    xmask = xindex < xnumel   # [1, XBLOCK]
    ymask = yindex < ynumel   # [YBLOCK, 1]
    y0 = yindex % 3           # [YBLOCK, 1] -- scalar constant per CTA when YBLOCK=1
    y1 = yindex // 3          # [YBLOCK, 1]
    # Load: broadcasts to [YBLOCK, XBLOCK]
    tmp0 = tl.load(in_ptr0 + (xindex + 50176 * yindex), mask=xmask & ymask, eviction_policy='evict_last')
    # Store: y0 + 3*xindex + 150528*y1 broadcasts to [YBLOCK, XBLOCK]
    tl.store(out_ptr0 + (y0 + 3 * xindex + 150528 * y1), tmp0, mask=xmask & ymask)

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
