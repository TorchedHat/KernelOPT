import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # 3D grid: pid(0) -> HW tiles, pid(1) -> C tiles, pid(2) -> batch
    # XBLOCK is used as TILE size for both dimensions (set to 32 in kernel_function)
    # YBLOCK is kept for signature compatibility but unused
    pid_x = tl.program_id(0)      # indexes HW dimension (31500)
    pid_y = tl.program_id(1)      # indexes C dimension (128)
    pid_batch = tl.program_id(2)  # indexes batch (32)

    # Tile index vectors — use XBLOCK directly as constexpr for tl.arange
    x2_idx = pid_x * XBLOCK + tl.arange(0, XBLOCK)  # HW indices [XBLOCK]
    y0_idx = pid_y * XBLOCK + tl.arange(0, XBLOCK)  # C indices [XBLOCK]

    # Boundary masks
    x2_mask = x2_idx < 31500  # HW=31500 is not a multiple of 32
    y0_mask = y0_idx < 128    # C=128 IS a multiple of 32, always true but keep for safety

    # Load: shape [XBLOCK_x2, XBLOCK_y0] — x2 varies along rows, y0 varies along cols (stride-1 in input -> coalesced)
    x2_2d = x2_idx[:, None]   # [XBLOCK, 1]
    y0_2d = y0_idx[None, :]   # [1, XBLOCK]
    mask_load = x2_mask[:, None] & y0_mask[None, :]
    load_idx = y0_2d + 128 * x2_2d + 4032000 * pid_batch
    tile = tl.load(in_ptr0 + load_idx, mask=mask_load, other=0.0)

    # Transpose in registers: [XBLOCK_x2, XBLOCK_y0] -> [XBLOCK_y0, XBLOCK_x2]
    tile_T = tl.trans(tile)

    # Store: shape [XBLOCK_y0, XBLOCK_x2] — x2 varies along cols (stride-1 in output -> coalesced)
    y0_out = y0_idx[:, None]   # [XBLOCK, 1]
    x2_out = x2_idx[None, :]   # [1, XBLOCK]
    mask_store = y0_mask[:, None] & x2_mask[None, :]
    store_idx = x2_out + 31500 * (y0_out + 128 * pid_batch)
    tl.store(out_ptr0 + store_idx, tile_T, mask=mask_store)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
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
