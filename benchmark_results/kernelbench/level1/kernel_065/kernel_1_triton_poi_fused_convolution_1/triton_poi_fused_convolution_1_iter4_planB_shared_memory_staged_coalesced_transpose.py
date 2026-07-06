import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 64, 'YBLOCK': 64}, num_warps=16, num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 31500
    # Use XBLOCK as the tile size for both dimensions (square tiles via autotune)
    TILE: tl.constexpr = XBLOCK
    pid_x = tl.program_id(0)  # covers xnumel (H*W=31500) dimension
    pid_y = tl.program_id(1)  # covers ynumel (batch*channel=4096) dimension
    x_offs = pid_x * TILE + tl.arange(0, TILE)  # [TILE]
    y_offs = pid_y * TILE + tl.arange(0, TILE)  # [TILE]
    xmask = x_offs < xnumel
    ymask = y_offs < ynumel
    y0 = y_offs % 128  # stride-1 dim in source tensor (channel dim, contiguous)
    y1 = y_offs // 128  # batch dim
    # Load [TILE_x, TILE_y] with y0 (stride-1) as fast (inner) dim -> COALESCED
    x2d = x_offs[:, None]    # [TILE, 1]
    y0_2d = y0[None, :]      # [1, TILE]
    y1_2d = y1[None, :]      # [1, TILE]
    load_mask = xmask[:, None] & ymask[None, :]  # [TILE_x, TILE_y]
    tile_data = tl.load(in_ptr0 + (y0_2d + 128*x2d + 4032000*y1_2d), mask=load_mask, other=0.0, eviction_policy='evict_first')  # [TILE_x, TILE_y] coalesced along y0
    # Transpose in registers: [TILE_x, TILE_y] -> [TILE_y, TILE_x]
    tile_T = tl.trans(tile_data)
    # Store [TILE_y, TILE_x] with x2 (stride-1 in dest) as fast (inner) dim -> COALESCED
    y3_2d = y_offs[:, None]      # [TILE_y, 1]
    x2d_store = x_offs[None, :]  # [1, TILE_x]
    store_mask = ymask[:, None] & xmask[None, :]  # [TILE_y, TILE_x]
    tl.store(out_ptr0 + (x2d_store + 31500*y3_2d), tile_T, mask=store_mask, eviction_policy='evict_first')  # coalesced along x2

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
