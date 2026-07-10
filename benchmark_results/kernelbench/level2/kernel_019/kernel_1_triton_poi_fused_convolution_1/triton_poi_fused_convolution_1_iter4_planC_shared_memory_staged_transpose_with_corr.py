import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 64, 'XBLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 16, 'XBLOCK': 16}, num_warps=4, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    # Use XBLOCK as the tile size for the x (spatial) dimension
    # Use YBLOCK as the tile size for the y (channel/batch) dimension
    # Both must be equal for the square-tile staged transpose
    x_tile_start = tl.program_id(0) * XBLOCK
    y_tile_start = tl.program_id(1) * YBLOCK
    # Load tile: lane_y is warp-fast (stride-1 in input)
    lane_x = tl.arange(0, XBLOCK)[:, None]   # [XBLOCK, 1] slow
    lane_y = tl.arange(0, YBLOCK)[None, :]   # [1, YBLOCK] fast
    x2 = x_tile_start + lane_x               # spatial index [XBLOCK, 1]
    y3 = y_tile_start + lane_y               # full y-index  [1, YBLOCK]
    y0 = y3 % 128    # channel (stride-1 in input)
    y1 = y3 // 128   # batch group
    load_mask = (x2 < xnumel) & (y3 < ynumel)
    # COALESCED LOAD: consecutive threads vary y (stride-1 in input)
    data = tl.load(in_ptr0 + (y0 + 128 * x2 + 4032000 * y1), mask=load_mask, other=0.0)
    # Transpose in-register: data[XBLOCK, YBLOCK] -> data_T[YBLOCK, XBLOCK]
    data_T = tl.trans(data)
    # Store tile: store_x2 is warp-fast (stride-1 in output)
    store_x2 = x_tile_start + tl.arange(0, XBLOCK)[None, :]  # [1, XBLOCK] fast
    store_y3 = y_tile_start + tl.arange(0, YBLOCK)[:, None]  # [YBLOCK, 1] slow
    store_mask = (store_x2 < xnumel) & (store_y3 < ynumel)
    # COALESCED STORE: consecutive threads vary x2 (stride-1 in output)
    tl.store(out_ptr0 + (store_x2 + 31500 * store_y3), data_T, mask=store_mask)

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
