import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # XBLOCK covers channel y0 (stride=1 in in_ptr0) -> COALESCED loads
    # YBLOCK covers spatial*batch (x2, y1)
    ch_offset = tl.program_id(0) * XBLOCK
    ch_index = ch_offset + tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK]
    ch_mask = ch_index < 96
    sb_offset = tl.program_id(1) * YBLOCK
    sb_index = sb_offset + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    sb_mask = sb_index < (256 * 3025)
    x2 = sb_index % 3025   # spatial index
    y1 = sb_index // 3025  # batch index
    y0 = ch_index          # channel index (innermost, stride=1)
    y3 = y1 * 96 + y0     # flattened batch*channel
    mask = ch_mask & sb_mask
    # LOAD: consecutive threads vary y0 (channel, stride=1) -> COALESCED
    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), mask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + y0, mask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    # STORE: address x2 + 3025*y3; consecutive threads vary y0 with stride 3025
    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, mask)

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
