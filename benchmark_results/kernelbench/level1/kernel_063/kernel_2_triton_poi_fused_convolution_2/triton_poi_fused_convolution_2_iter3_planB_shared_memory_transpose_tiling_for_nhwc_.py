import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # YBLOCK=CBLOCK (channel), XBLOCK=SBLOCK (spatial)
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # YBLOCK = CBLOCK (channel dimension, stride-1 in in_ptr0 -> coalesced load)
    # XBLOCK = SBLOCK (spatial dimension, stride-1 in out_ptr0 -> coalesced store)
    c_pid = tl.program_id(0)   # channel tile
    s_pid = tl.program_id(1)   # spatial tile
    b_pid = tl.program_id(2)   # batch index
    c_off = c_pid * YBLOCK + tl.arange(0, YBLOCK)   # [YBLOCK] channel indices (stride-1 in in_ptr0)
    s_off = s_pid * XBLOCK + tl.arange(0, XBLOCK)   # [XBLOCK] spatial indices (stride-1 in out_ptr0)
    c_mask = c_off < 96    # channel boundary mask
    s_mask = s_off < 3025  # spatial boundary mask
    # LOAD: shape [XBLOCK, YBLOCK] - inner dim is channel (stride-1) -> COALESCED
    load_addr = b_pid * 290400 + s_off[:, None] * 96 + c_off[None, :]  # [XBLOCK, YBLOCK]
    load_mask = s_mask[:, None] & c_mask[None, :]
    tile_in = tl.load(in_ptr0 + load_addr, mask=load_mask, other=0.0, eviction_policy='evict_first')
    # BIAS: shape [YBLOCK], broadcast over spatial
    bias = tl.load(in_ptr1 + c_off, mask=c_mask, other=0.0, eviction_policy='evict_last')  # [YBLOCK]
    tile_out = tile_in + bias[None, :]  # [XBLOCK, YBLOCK]
    # STORE: need [YBLOCK, XBLOCK] - inner dim is spatial (stride-1 in NCHW) -> COALESCED
    # tl.trans() converts [XBLOCK, YBLOCK] -> [YBLOCK, XBLOCK] in-register (no smem needed)
    store_addr = b_pid * 290400 + c_off[:, None] * 3025 + s_off[None, :]  # [YBLOCK, XBLOCK]
    store_mask = c_mask[:, None] & s_mask[None, :]
    tl.store(out_ptr0 + store_addr, tl.trans(tile_out), mask=store_mask, eviction_policy='evict_first')

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
