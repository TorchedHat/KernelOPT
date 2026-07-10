import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'YBLOCK': 512},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 1024}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 256},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8,   'YBLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 256},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8,   'YBLOCK': 256},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 4,   'YBLOCK': 256},  num_warps=2, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # xnumel_ch=30 (channel dim), ynumel_sp=50176 (spatial dim) - SWAPPED from original
    xnumel_ch = 30
    ynumel_sp = 50176
    # x-tile covers channel dimension -> store will be stride-1 (contiguous)
    xoffset = tl.program_id(0) * XBLOCK
    xrange  = tl.arange(0, XBLOCK)
    xrange  = tl.max_contiguous(tl.multiple_of(xrange, XBLOCK), XBLOCK)
    xindex  = xoffset + xrange[None, :]   # [1, XBLOCK] - channel index (0..29)
    xmask   = xindex < xnumel_ch
    # y-tile covers spatial dimension
    yoffset = tl.program_id(1) * YBLOCK
    yindex  = yoffset + tl.arange(0, YBLOCK)[:, None]  # [YBLOCK, 1] - spatial (0..50175)
    ymask   = yindex < ynumel_sp
    mask    = xmask & ymask
    # Decompose channel index into y0=channel%3 (in-group channel), y1=batch
    y0 = xindex % 3    # [1, XBLOCK] - channel within RGB group
    y1 = xindex // 3   # [1, XBLOCK] - batch index (0..9)
    x2 = yindex        # [YBLOCK, 1] - spatial pixel index
    # Load from NCHW: in_ptr0[spatial + 50176 * channel_flat]
    tmp0 = tl.load(in_ptr0 + (x2 + 50176 * xindex), mask=mask, other=0.0)
    # Store to NHWC: out_ptr0[channel%3 + 3*spatial + 150528*batch]
    # Adjacent threads in XBLOCK differ by 1 in xindex -> y0 differs by 1 -> stride-1 store!
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, mask=mask)

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
