import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=8, num_stages=3),
    ],
    key=['xnumel', 'ynumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # After axis swap: X tiles over batch*channel (24576), Y tiles over spatial (3025)
    xnumel = 24576
    ynumel = 3025
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    ymask = yindex < ynumel
    # c = channel index (0..95), n = batch index (0..255)
    c = xindex % 96    # stride=1 in NHWC input => coalesced loads
    n = xindex // 96   # batch index
    s = yindex         # spatial index (0..3024)
    # Load from NHWC input: in_ptr0 has strides (290400, 1, 5280, 96)
    # Flat index: n*290400 + c*1 + (s//55)*5280 + (s%55)*96
    # But since s = h*55 + w, and stride_h=5280=55*96, stride_w=96:
    # flat = n*290400 + c + s*96  (because h*5280 + w*96 = (h*55+w)*96 = s*96)
    tmp0 = tl.load(in_ptr0 + (c + 96*s + 290400*n), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (c), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    # Store to NCHW output: out_ptr0 has strides (290400, 3025, 55, 1)
    # Flat index: n*290400 + c*3025 + s
    tl.store(out_ptr0 + (s + 3025*c + 290400*n), tmp2, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((256, 96, 55, 55), (290400, 1, 5280, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256, 96, 55, 55), (290400, 3025, 55, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
