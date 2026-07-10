import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Reinterpret: XBLOCK = TILE_HW, YBLOCK = TILE_C
    # Grid: (cdiv(HW, TILE_HW), cdiv(C, TILE_C), N)
    # N=32, C=128, HW=31500
    # Input strides: (4032000, 1, 32000, 128) -> stride_in_n=4032000, stride_in_hw=128 (C innermost)
    # Output strides: (4032000, 31500, 250, 1) -> stride_out_n=4032000, stride_out_c=31500 (HW innermost)
    N = 32
    C = 128
    HW = 31500
    stride_in_n = 4032000
    stride_in_hw = 128
    stride_out_c = 31500

    pid_hw = tl.program_id(0)   # tiles over HW
    pid_c  = tl.program_id(1)   # tiles over C
    pid_n  = tl.program_id(2)   # tiles over N

    hw_off = pid_hw * XBLOCK
    c_off  = pid_c  * YBLOCK

    hw_range = hw_off + tl.arange(0, XBLOCK)  # [XBLOCK] = [TILE_HW]
    c_range  = c_off  + tl.arange(0, YBLOCK)  # [YBLOCK] = [TILE_C]

    hw_mask = hw_range < HW
    c_mask  = c_range  < C

    # COALESCED LOAD: inner dim = C (stride-1 in input)
    in_base = pid_n * stride_in_n
    in_ptrs = (in_base
               + hw_range[:, None] * stride_in_hw   # [XBLOCK, 1]
               + c_range[None, :])                  # [1, YBLOCK]
    load_mask = hw_mask[:, None] & c_mask[None, :]
    tile = tl.load(in_ptr0 + in_ptrs, mask=load_mask, other=0.0)  # [XBLOCK, YBLOCK] = [TILE_HW, TILE_C]

    # IN-REGISTER TRANSPOSE
    tile_t = tl.trans(tile)  # [YBLOCK, XBLOCK] = [TILE_C, TILE_HW]

    # COALESCED STORE: inner dim = HW (stride-1 in output)
    out_base = pid_n * stride_in_n
    out_ptrs = (out_base
                + c_range[:, None] * stride_out_c   # [YBLOCK, 1]
                + hw_range[None, :])                # [1, XBLOCK]
    store_mask = c_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr0 + out_ptrs, tile_t, mask=store_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 128, 126, 250), (4032000, 1, 32000, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 128, 126, 250), (4032000, 31500, 250, 1), device='cuda:0', dtype=torch.float32)
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
