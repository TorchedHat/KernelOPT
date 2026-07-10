import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # XBLOCK = TILE_NC (NC dimension, stride-1 in channels-last input)
    # YBLOCK = TILE_HW (spatial HW dimension, stride-1 in NCHW output)
    nc_offset = tl.program_id(0) * XBLOCK
    hw_offset = tl.program_id(1) * YBLOCK
    nc_idx = nc_offset + tl.arange(0, XBLOCK)  # [XBLOCK] = [TILE_NC]
    hw_idx = hw_offset + tl.arange(0, YBLOCK)  # [YBLOCK] = [TILE_HW]
    # Load phase: NC fast (warp-fast), HW slow -> coalesced loads from channels-last input
    nc_2d = nc_idx[None, :]   # [1, XBLOCK]
    hw_2d = hw_idx[:, None]   # [YBLOCK, 1]
    channel = nc_2d % 128     # stride-1 in input -> coalesced
    batch   = nc_2d // 128
    spatial = hw_2d
    load_mask = (nc_2d < 4096) & (hw_2d < 31500)
    vals = tl.load(in_ptr0 + (channel + 128*spatial + 4032000*batch), load_mask, other=0.0)
    # vals shape: [YBLOCK, XBLOCK] = [TILE_HW, TILE_NC]
    # Store phase: HW fast (warp-fast), NC slow -> coalesced stores to NCHW output
    hw_store = hw_idx[None, :]   # [1, YBLOCK] fast dim, stride=1 in output
    nc_store = nc_idx[:, None]   # [XBLOCK, 1] slow dim, stride=31500 in output
    store_mask = (nc_store < 4096) & (hw_store < 31500)
    # tl.trans(vals): [YBLOCK, XBLOCK] -> [XBLOCK, YBLOCK] = [TILE_NC, TILE_HW]
    # store index hw_store + 31500*nc_store: hw_store is fast -> COALESCED
    tl.store(out_ptr0 + (hw_store + 31500*nc_store), tl.trans(vals), store_mask)

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
