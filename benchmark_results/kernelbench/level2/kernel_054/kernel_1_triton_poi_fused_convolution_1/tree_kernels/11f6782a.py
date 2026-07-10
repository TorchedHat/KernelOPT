import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32,  'XBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 16},  num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 8},   num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 32,  'XBLOCK': 64},  num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 32},  num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 16},  num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 16,  'XBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 32,  'XBLOCK': 128}, num_warps=8, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # YBLOCK = TILE_C (NC dimension, ynumel=4096, stride-1 in input)
    # XBLOCK = TILE_HW (spatial HW dimension, xnumel=31500, stride-1 in output)
    pid_c  = tl.program_id(0)   # NC tiles
    pid_hw = tl.program_id(1)   # HW tiles

    # NC indices: channel = nc_idx % 128 (stride-1 in input), batch = nc_idx // 128
    nc_off = pid_c * YBLOCK
    nc_idx = nc_off + tl.arange(0, YBLOCK)   # shape [YBLOCK] = [TILE_C]
    nc_mask = nc_idx < ynumel
    channel = nc_idx % 128    # stride-1 in input -> coalesced load (warp-fast)
    batch   = nc_idx // 128

    # HW indices: h = hw_idx // 250, w = hw_idx % 250
    hw_off = pid_hw * XBLOCK
    hw_idx = hw_off + tl.arange(0, XBLOCK)   # shape [XBLOCK] = [TILE_HW]
    hw_mask = hw_idx < xnumel
    h = hw_idx // 250
    w = hw_idx % 250

    # Load tile: shape [XBLOCK, YBLOCK] = [TILE_HW, TILE_C]
    # Inner dim (YBLOCK / channel) is stride-1 in input -> COALESCED LOADS
    load_mask = hw_mask[:, None] & nc_mask[None, :]
    load_addr = (channel[None, :] + 128 * w[:, None] + 32000 * h[:, None] + 4032000 * batch[None, :])
    tile = tl.load(in_ptr0 + load_addr, mask=load_mask, other=0.0)

    # Transpose tile: [XBLOCK, YBLOCK] -> [YBLOCK, XBLOCK] = [TILE_C, TILE_HW]
    tile_T = tl.trans(tile)

    # Store tile: shape [YBLOCK, XBLOCK] = [TILE_C, TILE_HW]
    # Inner dim (XBLOCK / hw_idx) is stride-1 in output -> COALESCED STORES
    store_mask = nc_mask[:, None] & hw_mask[None, :]
    store_addr = hw_idx[None, :] + 31500 * nc_idx[:, None]
    tl.store(out_ptr0 + store_addr, tile_T, mask=store_mask)

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
