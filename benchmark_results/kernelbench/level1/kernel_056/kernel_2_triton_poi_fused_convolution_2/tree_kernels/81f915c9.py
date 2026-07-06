import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # XBLOCK = TILE_C (channel dim, stride-1 loads), YBLOCK = TILE_S (spatial dim, stride-1 stores)
        # All values must be powers of 2 for tl.arange
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=8, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Reinterpret axes for coalesced access:
    #   XBLOCK (pid(1)) -> channel*batch tiles: stride-1 in input -> coalesced loads
    #   YBLOCK (pid(0)) -> spatial tiles: stride-1 in output -> coalesced stores
    # ynumel = 24576 = 256*96 (N*C), xnumel = 3025 (S = 55*55)
    ynumel = 24576
    xnumel = 3025

    # pid(0) -> spatial tiles (YBLOCK threads, stride-1 in output)
    pid_s  = tl.program_id(0)
    # pid(1) -> channel*batch tiles (XBLOCK threads, stride-1 in input)
    pid_nc = tl.program_id(1)

    s_off  = pid_s  * YBLOCK + tl.arange(0, YBLOCK)   # spatial offsets [YBLOCK]
    nc_off = pid_nc * XBLOCK + tl.arange(0, XBLOCK)   # NC offsets [XBLOCK]

    c_idx = nc_off % 96     # channel index [0, 96)
    n_idx = nc_off // 96    # batch index [0, 256)

    s_mask  = s_off  < xnumel   # xnumel = 3025
    nc_mask = nc_off < ynumel   # ynumel = 24576

    # Load tile: shape [YBLOCK, XBLOCK]
    # Input strides (290400, 1, 5280, 96): addr = c_idx + 96*s_off + 290400*n_idx
    # c_idx varies along XBLOCK (innermost, stride-1) -> coalesced loads
    load_addr = c_idx[None, :] + 96 * s_off[:, None] + 290400 * n_idx[None, :]
    load_mask = s_mask[:, None] & nc_mask[None, :]
    tile = tl.load(in_ptr0 + load_addr, mask=load_mask, other=0.0)

    # Bias: shape [XBLOCK], broadcast over YBLOCK
    bias = tl.load(in_ptr1 + c_idx, mask=nc_mask, other=0.0)
    tile = tile + bias[None, :]

    # Store tile: shape [YBLOCK, XBLOCK]
    # Output strides (290400, 3025, 55, 1): addr = s_off + 3025*c_idx + 290400*n_idx
    # s_off varies along YBLOCK (innermost in store, stride-1) -> coalesced stores
    store_addr = s_off[:, None] + 3025 * c_idx[None, :] + 290400 * n_idx[None, :]
    tl.store(out_ptr0 + store_addr, tile, mask=load_mask)

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
