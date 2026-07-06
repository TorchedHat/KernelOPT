import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32,  'XBLOCK': 32},  num_warps=4,  num_stages=2),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 32},  num_warps=8,  num_stages=2),
        triton.Config({'YBLOCK': 32,  'XBLOCK': 64},  num_warps=4,  num_stages=2),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 64},  num_warps=8,  num_stages=2),
        triton.Config({'YBLOCK': 32,  'XBLOCK': 32},  num_warps=8,  num_stages=3),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 32},  num_warps=16, num_stages=2),
        triton.Config({'YBLOCK': 32,  'XBLOCK': 64},  num_warps=8,  num_stages=3),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 64},  num_warps=16, num_stages=3),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 32},  num_warps=8,  num_stages=2),
        triton.Config({'YBLOCK': 32,  'XBLOCK': 128}, num_warps=8,  num_stages=2),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 64},  num_warps=16, num_stages=2),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 128}, num_warps=16, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # YBLOCK = CBLOCK (channel tile, warp-fast for loads)
    # XBLOCK = SBLOCK (spatial tile, warp-fast for stores)
    ynumel = 24576
    xnumel = 3025
    C = 96

    c_off = tl.program_id(1) * YBLOCK  # channel*batch tile offset
    s_off = tl.program_id(0) * XBLOCK  # spatial tile offset

    # YBLOCK is the warp-fast (innermost) dimension for LOADS (stride-1 channel in NHWC)
    c_idx = c_off + tl.arange(0, YBLOCK)[None, :]  # shape [1, YBLOCK]
    s_idx = s_off + tl.arange(0, XBLOCK)[:, None]  # shape [XBLOCK, 1]

    c_mask = c_idx < ynumel
    s_mask = s_idx < xnumel
    mask = c_mask & s_mask

    y0 = c_idx % C      # channel: stride-1 in NHWC input -> COALESCED along YBLOCK
    y1 = c_idx // C     # batch index

    # PHASE 1: COALESCED LOAD from NHWC input
    # Address: y0 + 96*s_idx + 290400*y1, shape [XBLOCK, YBLOCK]
    # Consecutive threads vary over c_idx (YBLOCK fast dim) -> y0 varies -> stride-1 -> COALESCED
    load_addr = y0 + C * s_idx + 290400 * y1
    tmp0 = tl.load(in_ptr0 + load_addr, mask, eviction_policy='evict_first')

    # Bias: shape [1, YBLOCK], broadcast over XBLOCK rows — reused from L1
    tmp1 = tl.load(in_ptr1 + y0, c_mask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1  # [XBLOCK, YBLOCK]

    # PHASE 2: In-register transpose then COALESCED STORE to NCHW output
    tmp2_T = tl.trans(tmp2)  # [YBLOCK, XBLOCK] — transpose in registers

    # For store: XBLOCK is the warp-fast dimension -> stride-1 in NCHW output -> COALESCED
    c_idx_st = c_off + tl.arange(0, YBLOCK)[:, None]  # shape [YBLOCK, 1]
    s_idx_st = s_off + tl.arange(0, XBLOCK)[None, :]  # shape [1, XBLOCK]
    c_mask_st = c_idx_st < ynumel
    s_mask_st = s_idx_st < xnumel
    mask_st = c_mask_st & s_mask_st

    # Address: s_idx_st + 3025*c_idx_st, shape [YBLOCK, XBLOCK]
    # Consecutive threads vary over s_idx_st (XBLOCK fast dim) -> stride-1 -> COALESCED
    store_addr = s_idx_st + 3025 * c_idx_st
    tl.store(out_ptr0 + store_addr, tmp2_T, mask_st, eviction_policy='evict_first')

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
