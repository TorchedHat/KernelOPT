import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 96, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 96, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 96, 'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 48, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 48, 'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 96, 'YBLOCK': 32}, num_warps=8, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # xnumel = 24576 (batch*channel = 256*96) — contiguous axis in in_ptr0 (stride=1)
    # ynumel = 3025  (spatial = 55*55)        — strided axis in in_ptr0 (stride=96)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    ymask = yindex < ynumel
    # Decompose xindex into channel c (stride=1 in in_ptr0) and batch b
    c = xindex % 96
    b = xindex // 96
    s = yindex  # spatial index (stride=96 in in_ptr0)
    # Vectorization hints: channel axis is contiguous and aligned to XBLOCK
    # XBLOCK is always a divisor of 96 in our configs (96, 48, 32),
    # so c = xoffset%96 + arange(0, XBLOCK) is a contiguous run aligned to XBLOCK
    c_hint = tl.max_contiguous(tl.multiple_of(c, XBLOCK), XBLOCK)
    # Load from NHWC in_ptr0: address = b*290400 + c*1 + s*96
    # c is the fast axis (stride=1) -> enables 128-bit vectorized LDG.128
    tmp0 = tl.load(in_ptr0 + (c_hint + 96*s + 290400*b), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + c_hint, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    # Store to NCHW out_ptr0: address = b*290400 + c*3025 + s*1
    # s is the fast axis (stride=1) -> coalesced store
    tl.store(out_ptr0 + (s + 3025*(c_hint + 96*b)), tmp2, xmask & ymask)

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
