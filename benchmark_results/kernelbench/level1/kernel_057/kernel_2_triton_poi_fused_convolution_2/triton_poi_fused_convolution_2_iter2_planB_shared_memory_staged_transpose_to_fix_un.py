import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=4, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # TILE_HW = XBLOCK (spatial tile), TILE_C = YBLOCK (channel tile)
    # 3D grid: pid(0)=hw tile, pid(1)=channel tile, pid(2)=batch
    pid_hw = tl.program_id(0)   # spatial tile
    pid_c  = tl.program_id(1)   # channel tile
    pid_n  = tl.program_id(2)   # batch index

    hw_off = pid_hw * XBLOCK
    c_off  = pid_c  * YBLOCK

    hw_range = hw_off + tl.arange(0, XBLOCK)   # [TILE_HW]
    c_range  = c_off  + tl.arange(0, YBLOCK)   # [TILE_C]

    hw_mask = hw_range < 3025
    c_mask  = c_range  < 96

    # COALESCED LOAD: inner dim = C (stride=1 in NHWC input)
    load_mask = hw_mask[:, None] & c_mask[None, :]  # [TILE_HW, TILE_C]
    load_addr = pid_n * 290400 + hw_range[:, None] * 96 + c_range[None, :]
    tile = tl.load(in_ptr0 + load_addr, mask=load_mask, other=0.0)  # [TILE_HW, TILE_C]

    # Bias add (broadcast over HW)
    bias = tl.load(in_ptr1 + c_range, mask=c_mask, other=0.0)  # [TILE_C]
    tile = tile + bias[None, :]  # [TILE_HW, TILE_C]

    # In-register transpose
    tile_T = tl.trans(tile)  # [TILE_C, TILE_HW]

    # COALESCED STORE: inner dim = HW (stride=1 in NCHW output)
    store_mask = c_mask[:, None] & hw_mask[None, :]  # [TILE_C, TILE_HW]
    store_addr = pid_n * 290400 + c_range[:, None] * 3025 + hw_range[None, :]
    tl.store(out_ptr0 + store_addr, tile_T, mask=store_mask)

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
