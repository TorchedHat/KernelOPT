import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # TILE_C = channel tile size (fast dim for loads, stride=1 in NHWC input)
    # TILE_HW = spatial tile size (fast dim for stores, stride=1 in NCHW output)
    TILE_C: tl.constexpr = 32
    TILE_HW: tl.constexpr = 32
    C: tl.constexpr = 96
    HW: tl.constexpr = 3025
    N_C_TILES: tl.constexpr = C // TILE_C  # = 3

    hw_tile = tl.program_id(0)
    nc_tile = tl.program_id(1)

    c_tile = nc_tile % N_C_TILES
    n = nc_tile // N_C_TILES

    c_base = c_tile * TILE_C
    hw_base = hw_tile * TILE_HW

    # Linearize threads: TILE_C * TILE_HW = 1024 threads per CTA
    # tc is fast (channel, stride=1 in NHWC input for coalesced load)
    # thw is slow (spatial)
    tid = tl.arange(0, TILE_C * TILE_HW)
    tc = tid % TILE_C    # fast: channel index within tile
    thw = tid // TILE_C  # slow: spatial index within tile

    c_idx = c_base + tc
    hw_idx = hw_base + thw

    load_mask = (c_idx < C) & (hw_idx < HW)

    # Load from NHWC input: stride is (290400, 1, 5280, 96)
    # index = n*290400 + hw_idx*96 + c_idx  (hw_idx*96 because H*W stride=96 in NHWC? No)
    # Actually NHWC: n*290400 + h*5280 + w*96 + c
    # hw_idx = h*55 + w, so h = hw_idx // 55, w = hw_idx % 55
    # index = n*290400 + (hw_idx//55)*5280 + (hw_idx%55)*96 + c_idx
    # But simpler: stride for spatial in NHWC is 96 (W stride) and 5280 (H stride)
    # Since hw_idx = h*55+w: index = n*290400 + hw_idx*96 + c_idx
    # Wait: h*5280 + w*96 = h*55*96 + w*96 = (h*55+w)*96 = hw_idx*96. Yes!
    load_idx = n * 290400 + hw_idx * 96 + c_idx
    tmp0 = tl.load(in_ptr0 + load_idx, mask=load_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + c_idx, mask=(c_idx < C), other=0.0)
    tmp2 = tmp0 + tmp1

    # Now store to NCHW output: strides (290400, 3025, 55, 1)
    # index = n*290400 + c_idx*3025 + hw_idx
    # For coalesced store, we want hw to be fast-varying.
    # Currently thw is slow and tc is fast — this means c_idx varies fast in store,
    # which gives stride-3025 stores (uncoalesced).
    # To fix: we need to transpose so hw_idx varies fast.
    # We do this by re-mapping tid for the store phase:
    # tc2 = tid // TILE_HW  (slow channel for store)
    # thw2 = tid % TILE_HW  (fast spatial for store)
    # But we need to read tmp2 values in transposed order.
    # Since Triton doesn't support true shared memory scatter/gather,
    # we use tl.reshape + tl.permute to do the transpose in registers.

    # Reshape tmp2 from [TILE_C*TILE_HW] to [TILE_HW, TILE_C] (thw rows, tc cols)
    tmp2_2d = tl.reshape(tmp2, [TILE_HW, TILE_C])
    # Transpose to [TILE_C, TILE_HW]
    tmp2_t = tl.permute(tmp2_2d, [1, 0])
    # Flatten back to [TILE_C*TILE_HW]
    tmp2_flat = tl.reshape(tmp2_t, [TILE_C * TILE_HW])

    # Now tid maps as: tc2 = tid // TILE_HW (channel), thw2 = tid % TILE_HW (spatial)
    tid2 = tl.arange(0, TILE_C * TILE_HW)
    tc2 = tid2 // TILE_HW   # slow: channel
    thw2 = tid2 % TILE_HW   # fast: spatial (stride=1 in NCHW output)

    c2 = c_base + tc2
    hw2 = hw_base + thw2

    store_mask = (c2 < C) & (hw2 < HW)
    store_idx = n * 290400 + c2 * 3025 + hw2
    tl.store(out_ptr0 + store_idx, tmp2_flat, mask=store_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025, YBLOCK=8, XBLOCK=128)
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
