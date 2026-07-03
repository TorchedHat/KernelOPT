import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Reinterpret tile dims:
    #   YBLOCK = TILE_C  (channel axis, stride=1 in NHWC input)
    #   XBLOCK = TILE_HW (spatial axis, stride=1 in NCHW output)
    # Grid: pid(0)=hw tile, pid(1)=channel tile, pid(2)=batch
    # N=256, C=96, HW=3025
    pid_n  = tl.program_id(2)   # batch dimension
    pid_c  = tl.program_id(1)   # channel tile (YBLOCK wide)
    pid_hw = tl.program_id(0)   # spatial tile (XBLOCK wide)

    c_offs  = pid_c  * YBLOCK + tl.arange(0, YBLOCK)
    hw_offs = pid_hw * XBLOCK + tl.arange(0, XBLOCK)

    c_mask  = c_offs  < 96
    hw_mask = hw_offs < 3025

    # COALESCED LOAD: threads vary over c (stride=1 in NHWC input)
    # in_ptr0 flat index: n*290400 + hw*96 + c  (strides: 290400, 96, 1)
    load_idx = pid_n * 290400 + hw_offs[:, None] * 96 + c_offs[None, :]
    load_mask = hw_mask[:, None] & c_mask[None, :]
    tile = tl.load(in_ptr0 + load_idx, mask=load_mask, other=0.0)  # [XBLOCK, YBLOCK] = [TILE_HW, TILE_C]

    # ADD BIAS broadcast over HW
    bias = tl.load(in_ptr1 + c_offs, mask=c_mask, other=0.0)  # [YBLOCK] = [TILE_C]
    tile = tile + bias[None, :]  # [TILE_HW, TILE_C]

    # IN-REGISTER TRANSPOSE: [TILE_HW, TILE_C] -> [TILE_C, TILE_HW]
    tile_t = tl.trans(tile)  # [YBLOCK, XBLOCK] = [TILE_C, TILE_HW]

    # COALESCED STORE: threads vary over hw (stride=1 in NCHW output)
    # out_ptr0 flat index: n*290400 + c*3025 + hw  (strides: 290400, 3025, 1)
    store_idx = pid_n * 290400 + c_offs[:, None] * 3025 + hw_offs[None, :]
    store_mask = c_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr0 + store_idx, tile_t, mask=store_mask)

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
