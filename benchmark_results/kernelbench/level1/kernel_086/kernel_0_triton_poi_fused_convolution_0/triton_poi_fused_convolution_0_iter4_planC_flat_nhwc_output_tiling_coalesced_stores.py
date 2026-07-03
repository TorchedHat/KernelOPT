import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat NHWC output tiling: total elements = 10 * 3 * 224 * 224 = 1_505_280
    # YBLOCK is forced to 1 in all configs; we use XBLOCK as the flat tile size
    TOTAL: tl.constexpr = 1505280
    CHW: tl.constexpr = 150528  # 3 * 224 * 224
    HW: tl.constexpr  = 50176   # 224 * 224
    C: tl.constexpr   = 3

    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < TOTAL

    # Decompose flat NHWC output index into (n, hw, c)
    n   = xindex // CHW
    rem = xindex % CHW
    hw  = rem // C
    c   = rem % C

    # Compute NCHW input offset: n*CHW + c*HW + hw
    in_idx = n * CHW + c * HW + hw

    # Load from NCHW input (streaming)
    val = tl.load(in_ptr0 + in_idx, mask=xmask, other=0.0, eviction_policy='evict_first')

    # Store to flat NHWC output (stride-1, fully coalesced)
    tl.store(out_ptr0 + xindex, val, mask=xmask)

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
