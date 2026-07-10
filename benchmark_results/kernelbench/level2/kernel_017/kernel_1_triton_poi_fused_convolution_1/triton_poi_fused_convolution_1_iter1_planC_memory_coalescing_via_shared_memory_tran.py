import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 16, 'XBLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 32}, num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'ynumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # in_ptr0 layout: (32, 128, 126, 250) stride (4032000, 1, 32000, 128)
    #   -> channel dim (stride=1) is innermost in memory
    # out_ptr0 layout: (32, 128, 126, 250) stride (4032000, 31500, 250, 1)
    #   -> w dim (stride=1) is innermost in memory
    # ynumel = N*C = 4096, xnumel = H*W = 31500
    # YBLOCK = TILE_C (channel tile, must divide 128)
    # XBLOCK = TILE_W (spatial tile)

    pid_x = tl.program_id(0)  # tiles over xnumel (H*W)
    pid_y = tl.program_id(1)  # tiles over ynumel (N*C)

    # Channel tile: YBLOCK must divide 128 so tiles never span a batch boundary
    nc_base = pid_y * YBLOCK
    c_base = nc_base % 128
    n_base = nc_base // 128

    c_offs = tl.arange(0, YBLOCK)
    c_idx = c_base + c_offs          # channel indices within [0, 128)
    # Guard with nc_mask for safety at the ynumel boundary
    nc_mask = (nc_base + c_offs) < ynumel  # [YBLOCK]

    # Spatial (W) tile
    w_base = pid_x * XBLOCK
    w_offs = tl.arange(0, XBLOCK)
    w_idx = w_base + w_offs          # flat spatial indices [XBLOCK]
    w_mask = w_idx < xnumel          # [XBLOCK]

    # Combined 2D mask [YBLOCK, XBLOCK]
    mask2d = nc_mask[:, None] & w_mask[None, :]

    # COALESCED LOAD: channel axis has stride=1 in input
    # in_ptr0 offset: c + 128*w + 4032000*n
    # c_idx (YBLOCK dim) varies along rows; w_idx (XBLOCK dim) varies along cols
    # For coalesced reads: innermost thread dimension should map to stride-1 axis (c)
    # We load [YBLOCK, XBLOCK] tile: c_idx[:, None] is the fast-varying dim in memory
    load_idx = c_idx[:, None] + 128 * w_idx[None, :] + 4032000 * n_base
    tile = tl.load(in_ptr0 + load_idx, mask=mask2d, other=0.0)

    # COALESCED STORE: w axis has stride=1 in output
    # out_ptr0 offset: w + 31500*c + 4032000*n
    # w_idx (XBLOCK dim) is the fast-varying dim in memory for output
    store_idx = w_idx[None, :] + 31500 * c_idx[:, None] + 4032000 * n_base
    tl.store(out_ptr0 + store_idx, tile, mask=mask2d)

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
